import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_0_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_fc_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_fc_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_fc_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_fc_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_fc_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_fc_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_fc_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_fc_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_0_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_0_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stem_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stem_modules_1_modules_conv_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stem_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_stem_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_1_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stem_modules_2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_stem_modules_2_modules_conv_parameters_weight_ = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stem_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_stem_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_2_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_2_modules_bn_parameters_bias_ = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (None)
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (None)
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (None)
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (None)
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (None)
        x_23 = torch.nn.functional.relu(x_22, inplace=True)
        x_22 = None
        x_24 = torch.cat([x_8, x_11, x_14, x_17, x_20, x_23], dim=1)
        x_8 = x_11 = x_14 = x_17 = x_20 = x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_24 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = (None)
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        x_se = x_27.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_ = (None)
        hardsigmoid = torch.nn.functional.hardsigmoid(x_se_1, False)
        x_se_1 = None
        x_28 = x_27 * hardsigmoid
        x_27 = hardsigmoid = None
        x_29 = torch.nn.functional.max_pool2d(
            x_28, 3, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        x_28 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (None)
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (None)
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (None)
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (None)
        x_44 = torch.nn.functional.relu(x_43, inplace=True)
        x_43 = None
        x_45 = torch.cat([x_29, x_32, x_35, x_38, x_41, x_44], dim=1)
        x_29 = x_32 = x_35 = x_38 = x_41 = x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = (None)
        x_48 = torch.nn.functional.relu(x_47, inplace=True)
        x_47 = None
        x_se_2 = x_48.mean((2, 3), keepdim=True)
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_ = (None)
        hardsigmoid_1 = torch.nn.functional.hardsigmoid(x_se_3, False)
        x_se_3 = None
        x_49 = x_48 * hardsigmoid_1
        x_48 = hardsigmoid_1 = None
        x_50 = torch.nn.functional.max_pool2d(
            x_49, 3, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (None)
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (None)
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (None)
        x_59 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (None)
        x_62 = torch.nn.functional.relu(x_61, inplace=True)
        x_61 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.relu(x_64, inplace=True)
        x_64 = None
        x_66 = torch.cat([x_50, x_53, x_56, x_59, x_62, x_65], dim=1)
        x_50 = x_53 = x_56 = x_59 = x_62 = x_65 = None
        x_67 = torch.conv2d(
            x_66,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_66 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = (None)
        x_69 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (None)
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (None)
        x_75 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (None)
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        x_85 = torch.cat([x_69, x_72, x_75, x_78, x_81, x_84], dim=1)
        x_72 = x_75 = x_78 = x_81 = x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_ = (None)
        x_88 = torch.nn.functional.relu(x_87, inplace=True)
        x_87 = None
        x_89 = x_88 + x_69
        x_88 = x_69 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (None)
        x_92 = torch.nn.functional.relu(x_91, inplace=True)
        x_91 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (None)
        x_95 = torch.nn.functional.relu(x_94, inplace=True)
        x_94 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (None)
        x_98 = torch.nn.functional.relu(x_97, inplace=True)
        x_97 = None
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (None)
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (None)
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_105 = torch.cat([x_89, x_92, x_95, x_98, x_101, x_104], dim=1)
        x_92 = x_95 = x_98 = x_101 = x_104 = None
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_105 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_bias_ = (None)
        x_108 = torch.nn.functional.relu(x_107, inplace=True)
        x_107 = None
        x_109 = x_108 + x_89
        x_108 = x_89 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (None)
        x_115 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (None)
        x_118 = torch.nn.functional.relu(x_117, inplace=True)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (None)
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (None)
        x_124 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        x_125 = torch.cat([x_109, x_112, x_115, x_118, x_121, x_124], dim=1)
        x_112 = x_115 = x_118 = x_121 = x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_concat_modules_bn_parameters_bias_ = (None)
        x_128 = torch.nn.functional.relu(x_127, inplace=True)
        x_127 = None
        x_se_4 = x_128.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_fc_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_fc_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_fc_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_fc_parameters_bias_ = (None)
        hardsigmoid_2 = torch.nn.functional.hardsigmoid(x_se_5, False)
        x_se_5 = None
        x_129 = x_128 * hardsigmoid_2
        x_128 = hardsigmoid_2 = None
        x_130 = x_129 + x_109
        x_129 = x_109 = None
        x_131 = torch.nn.functional.max_pool2d(
            x_130, 3, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (None)
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (None)
        x_137 = torch.nn.functional.relu(x_136, inplace=True)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (None)
        x_140 = torch.nn.functional.relu(x_139, inplace=True)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (None)
        x_143 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (None)
        x_146 = torch.nn.functional.relu(x_145, inplace=True)
        x_145 = None
        x_147 = torch.cat([x_131, x_134, x_137, x_140, x_143, x_146], dim=1)
        x_131 = x_134 = x_137 = x_140 = x_143 = x_146 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = (None)
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (None)
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (None)
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (None)
        x_159 = torch.nn.functional.relu(x_158, inplace=True)
        x_158 = None
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (None)
        x_162 = torch.nn.functional.relu(x_161, inplace=True)
        x_161 = None
        x_163 = torch.conv2d(
            x_162,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_163 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (None)
        x_165 = torch.nn.functional.relu(x_164, inplace=True)
        x_164 = None
        x_166 = torch.cat([x_150, x_153, x_156, x_159, x_162, x_165], dim=1)
        x_153 = x_156 = x_159 = x_162 = x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_166 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_ = (None)
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        x_170 = x_169 + x_150
        x_169 = x_150 = None
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_172 = torch.nn.functional.batch_norm(
            x_171,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_171 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (None)
        x_173 = torch.nn.functional.relu(x_172, inplace=True)
        x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (None)
        x_176 = torch.nn.functional.relu(x_175, inplace=True)
        x_175 = None
        x_177 = torch.conv2d(
            x_176,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_177 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (None)
        x_179 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (None)
        x_182 = torch.nn.functional.relu(x_181, inplace=True)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (None)
        x_185 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        x_186 = torch.cat([x_170, x_173, x_176, x_179, x_182, x_185], dim=1)
        x_173 = x_176 = x_179 = x_182 = x_185 = None
        x_187 = torch.conv2d(
            x_186,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_186 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_concat_modules_bn_parameters_bias_ = (None)
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        x_se_6 = x_189.mean((2, 3), keepdim=True)
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_fc_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_fc_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_fc_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_fc_parameters_bias_ = (None)
        hardsigmoid_3 = torch.nn.functional.hardsigmoid(x_se_7, False)
        x_se_7 = None
        x_190 = x_189 * hardsigmoid_3
        x_189 = hardsigmoid_3 = None
        x_191 = x_190 + x_170
        x_190 = x_170 = None
        x_192 = torch.nn.functional.adaptive_avg_pool2d(x_191, 1)
        x_191 = None
        x_193 = x_192.flatten(1, -1)
        x_192 = None
        x_194 = torch.nn.functional.dropout(x_193, 0.0, False, False)
        x_193 = None
        x_195 = torch._C._nn.linear(
            x_194,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_194 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_195,)
