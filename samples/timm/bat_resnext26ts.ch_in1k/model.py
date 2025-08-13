import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.silu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.silu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stem_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_stem_modules_conv3_modules_conv_parameters_weight_ = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv3_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv3_modules_bn_parameters_bias_ = None
        x_8 = torch.nn.functional.silu(x_7, inplace=True)
        x_7 = None
        input_1 = torch.nn.functional.max_pool2d(
            x_8, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_8 = None
        x_9 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_11 = torch.nn.functional.silu(x_10, inplace=True)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        x_11 = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_14 = torch.nn.functional.silu(x_13, inplace=True)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        rp = torch.nn.functional.adaptive_max_pool2d(x_20, (8, 1))
        cp = torch.nn.functional.adaptive_max_pool2d(x_20, (1, 8))
        x_20 = None
        conv2d_7 = torch.conv2d(
            rp,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        rp = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = (None)
        view = conv2d_7.view(1, 2, 8, 8)
        conv2d_7 = None
        p = view.sigmoid()
        view = None
        conv2d_8 = torch.conv2d(
            cp,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cp = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = (None)
        view_1 = conv2d_8.view(1, 2, 8, 8)
        conv2d_8 = None
        q = view_1.sigmoid()
        view_1 = None
        sum_1 = p.sum(dim=3, keepdim=True)
        p_1 = p / sum_1
        p = sum_1 = None
        sum_2 = q.sum(dim=2, keepdim=True)
        q_1 = q / sum_2
        q = sum_2 = None
        view_2 = p_1.view(1, 2, 1, 8, 8)
        p_1 = None
        expand = view_2.expand(1, 2, 8, 8, 8)
        view_2 = None
        p_2 = expand.contiguous()
        expand = None
        p_3 = p_2.view(1, 16, 8, 8)
        p_2 = None
        view_4 = q_1.view(1, 2, 1, 8, 8)
        q_1 = None
        expand_1 = view_4.expand(1, 2, 8, 8, 8)
        view_4 = None
        q_2 = expand_1.contiguous()
        expand_1 = None
        q_3 = q_2.view(1, 16, 8, 8)
        q_2 = None
        x_21 = p_3.view(16, -1, 1, 1)
        p_3 = None
        eye = torch.eye(8, 8, dtype=torch.float32, device=device(type="cpu"))
        x_22 = x_21 * eye
        x_21 = eye = None
        x_23 = x_22.view(16, 8, 8, 8, 8)
        x_22 = None
        split = torch.functional.split(x_23, 1, dim=1)
        x_23 = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3]
        getitem_4 = split[4]
        getitem_5 = split[5]
        getitem_6 = split[6]
        getitem_7 = split[7]
        split = None
        x_24 = torch.cat(
            (
                getitem,
                getitem_1,
                getitem_2,
                getitem_3,
                getitem_4,
                getitem_5,
                getitem_6,
                getitem_7,
            ),
            dim=3,
        )
        getitem = (
            getitem_1
        ) = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = None
        split_1 = torch.functional.split(x_24, 1, dim=2)
        x_24 = None
        getitem_8 = split_1[0]
        getitem_9 = split_1[1]
        getitem_10 = split_1[2]
        getitem_11 = split_1[3]
        getitem_12 = split_1[4]
        getitem_13 = split_1[5]
        getitem_14 = split_1[6]
        getitem_15 = split_1[7]
        split_1 = None
        x_25 = torch.cat(
            (
                getitem_8,
                getitem_9,
                getitem_10,
                getitem_11,
                getitem_12,
                getitem_13,
                getitem_14,
                getitem_15,
            ),
            dim=4,
        )
        getitem_8 = (
            getitem_9
        ) = (
            getitem_10
        ) = getitem_11 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = None
        x_26 = x_25.view(1, 16, 64, 64)
        x_25 = None
        x_27 = q_3.view(16, -1, 1, 1)
        q_3 = None
        eye_1 = torch.eye(8, 8, dtype=torch.float32, device=device(type="cpu"))
        x_28 = x_27 * eye_1
        x_27 = eye_1 = None
        x_29 = x_28.view(16, 8, 8, 8, 8)
        x_28 = None
        split_2 = torch.functional.split(x_29, 1, dim=1)
        x_29 = None
        getitem_16 = split_2[0]
        getitem_17 = split_2[1]
        getitem_18 = split_2[2]
        getitem_19 = split_2[3]
        getitem_20 = split_2[4]
        getitem_21 = split_2[5]
        getitem_22 = split_2[6]
        getitem_23 = split_2[7]
        split_2 = None
        x_30 = torch.cat(
            (
                getitem_16,
                getitem_17,
                getitem_18,
                getitem_19,
                getitem_20,
                getitem_21,
                getitem_22,
                getitem_23,
            ),
            dim=3,
        )
        getitem_16 = (
            getitem_17
        ) = (
            getitem_18
        ) = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = None
        split_3 = torch.functional.split(x_30, 1, dim=2)
        x_30 = None
        getitem_24 = split_3[0]
        getitem_25 = split_3[1]
        getitem_26 = split_3[2]
        getitem_27 = split_3[3]
        getitem_28 = split_3[4]
        getitem_29 = split_3[5]
        getitem_30 = split_3[6]
        getitem_31 = split_3[7]
        split_3 = None
        x_31 = torch.cat(
            (
                getitem_24,
                getitem_25,
                getitem_26,
                getitem_27,
                getitem_28,
                getitem_29,
                getitem_30,
                getitem_31,
            ),
            dim=4,
        )
        getitem_24 = (
            getitem_25
        ) = (
            getitem_26
        ) = getitem_27 = getitem_28 = getitem_29 = getitem_30 = getitem_31 = None
        x_32 = x_31.view(1, 16, 64, 64)
        x_31 = None
        y = x_26.matmul(x_17)
        x_26 = x_17 = None
        y_1 = y.matmul(x_32)
        y = x_32 = None
        x_33 = torch.conv2d(
            y_1,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        y_1 = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        y_2 = torch.nn.functional.dropout2d(x_38, 0.2, False, False)
        x_38 = None
        x_39 = y_2 + x_14
        y_2 = x_14 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_42 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_44 = x_41 + x_43
        x_41 = x_43 = None
        input_2 = torch.nn.functional.silu(x_44, inplace=True)
        x_44 = None
        x_45 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_47 = torch.nn.functional.silu(x_46, inplace=True)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        x_47 = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_50 = torch.nn.functional.silu(x_49, inplace=True)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        rp_1 = torch.nn.functional.adaptive_max_pool2d(x_56, (8, 1))
        cp_1 = torch.nn.functional.adaptive_max_pool2d(x_56, (1, 8))
        x_56 = None
        conv2d_17 = torch.conv2d(
            rp_1,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        rp_1 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = (None)
        view_12 = conv2d_17.view(1, 2, 8, 8)
        conv2d_17 = None
        p_4 = view_12.sigmoid()
        view_12 = None
        conv2d_18 = torch.conv2d(
            cp_1,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cp_1 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = (None)
        view_13 = conv2d_18.view(1, 2, 8, 8)
        conv2d_18 = None
        q_4 = view_13.sigmoid()
        view_13 = None
        sum_3 = p_4.sum(dim=3, keepdim=True)
        p_5 = p_4 / sum_3
        p_4 = sum_3 = None
        sum_4 = q_4.sum(dim=2, keepdim=True)
        q_5 = q_4 / sum_4
        q_4 = sum_4 = None
        view_14 = p_5.view(1, 2, 1, 8, 8)
        p_5 = None
        expand_2 = view_14.expand(1, 2, 8, 8, 8)
        view_14 = None
        p_6 = expand_2.contiguous()
        expand_2 = None
        p_7 = p_6.view(1, 16, 8, 8)
        p_6 = None
        view_16 = q_5.view(1, 2, 1, 8, 8)
        q_5 = None
        expand_3 = view_16.expand(1, 2, 8, 8, 8)
        view_16 = None
        q_6 = expand_3.contiguous()
        expand_3 = None
        q_7 = q_6.view(1, 16, 8, 8)
        q_6 = None
        x_57 = p_7.view(16, -1, 1, 1)
        p_7 = None
        eye_2 = torch.eye(8, 8, dtype=torch.float32, device=device(type="cpu"))
        x_58 = x_57 * eye_2
        x_57 = eye_2 = None
        x_59 = x_58.view(16, 8, 8, 8, 8)
        x_58 = None
        split_4 = torch.functional.split(x_59, 1, dim=1)
        x_59 = None
        getitem_32 = split_4[0]
        getitem_33 = split_4[1]
        getitem_34 = split_4[2]
        getitem_35 = split_4[3]
        getitem_36 = split_4[4]
        getitem_37 = split_4[5]
        getitem_38 = split_4[6]
        getitem_39 = split_4[7]
        split_4 = None
        x_60 = torch.cat(
            (
                getitem_32,
                getitem_33,
                getitem_34,
                getitem_35,
                getitem_36,
                getitem_37,
                getitem_38,
                getitem_39,
            ),
            dim=3,
        )
        getitem_32 = (
            getitem_33
        ) = (
            getitem_34
        ) = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = None
        split_5 = torch.functional.split(x_60, 1, dim=2)
        x_60 = None
        getitem_40 = split_5[0]
        getitem_41 = split_5[1]
        getitem_42 = split_5[2]
        getitem_43 = split_5[3]
        getitem_44 = split_5[4]
        getitem_45 = split_5[5]
        getitem_46 = split_5[6]
        getitem_47 = split_5[7]
        split_5 = None
        x_61 = torch.cat(
            (
                getitem_40,
                getitem_41,
                getitem_42,
                getitem_43,
                getitem_44,
                getitem_45,
                getitem_46,
                getitem_47,
            ),
            dim=4,
        )
        getitem_40 = (
            getitem_41
        ) = (
            getitem_42
        ) = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = None
        x_62 = x_61.view(1, 16, 64, 64)
        x_61 = None
        x_63 = q_7.view(16, -1, 1, 1)
        q_7 = None
        eye_3 = torch.eye(8, 8, dtype=torch.float32, device=device(type="cpu"))
        x_64 = x_63 * eye_3
        x_63 = eye_3 = None
        x_65 = x_64.view(16, 8, 8, 8, 8)
        x_64 = None
        split_6 = torch.functional.split(x_65, 1, dim=1)
        x_65 = None
        getitem_48 = split_6[0]
        getitem_49 = split_6[1]
        getitem_50 = split_6[2]
        getitem_51 = split_6[3]
        getitem_52 = split_6[4]
        getitem_53 = split_6[5]
        getitem_54 = split_6[6]
        getitem_55 = split_6[7]
        split_6 = None
        x_66 = torch.cat(
            (
                getitem_48,
                getitem_49,
                getitem_50,
                getitem_51,
                getitem_52,
                getitem_53,
                getitem_54,
                getitem_55,
            ),
            dim=3,
        )
        getitem_48 = (
            getitem_49
        ) = (
            getitem_50
        ) = getitem_51 = getitem_52 = getitem_53 = getitem_54 = getitem_55 = None
        split_7 = torch.functional.split(x_66, 1, dim=2)
        x_66 = None
        getitem_56 = split_7[0]
        getitem_57 = split_7[1]
        getitem_58 = split_7[2]
        getitem_59 = split_7[3]
        getitem_60 = split_7[4]
        getitem_61 = split_7[5]
        getitem_62 = split_7[6]
        getitem_63 = split_7[7]
        split_7 = None
        x_67 = torch.cat(
            (
                getitem_56,
                getitem_57,
                getitem_58,
                getitem_59,
                getitem_60,
                getitem_61,
                getitem_62,
                getitem_63,
            ),
            dim=4,
        )
        getitem_56 = (
            getitem_57
        ) = (
            getitem_58
        ) = getitem_59 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = None
        x_68 = x_67.view(1, 16, 64, 64)
        x_67 = None
        y_3 = x_62.matmul(x_53)
        x_62 = x_53 = None
        y_4 = y_3.matmul(x_68)
        y_3 = x_68 = None
        x_69 = torch.conv2d(
            y_4,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        y_4 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        y_5 = torch.nn.functional.dropout2d(x_74, 0.2, False, False)
        x_74 = None
        x_75 = y_5 + x_50
        y_5 = x_50 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_78 = x_77 + input_2
        x_77 = input_2 = None
        input_3 = torch.nn.functional.silu(x_78, inplace=True)
        x_78 = None
        x_79 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_81 = torch.nn.functional.silu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            4,
        )
        x_81 = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.silu(x_83, inplace=True)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_87 = torch.nn.functional.relu(x_86, inplace=True)
        x_86 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        rp_2 = torch.nn.functional.adaptive_max_pool2d(x_90, (8, 1))
        cp_2 = torch.nn.functional.adaptive_max_pool2d(x_90, (1, 8))
        x_90 = None
        conv2d_26 = torch.conv2d(
            rp_2,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        rp_2 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = (None)
        view_24 = conv2d_26.view(1, 2, 8, 8)
        conv2d_26 = None
        p_8 = view_24.sigmoid()
        view_24 = None
        conv2d_27 = torch.conv2d(
            cp_2,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cp_2 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = (None)
        view_25 = conv2d_27.view(1, 2, 8, 8)
        conv2d_27 = None
        q_8 = view_25.sigmoid()
        view_25 = None
        sum_5 = p_8.sum(dim=3, keepdim=True)
        p_9 = p_8 / sum_5
        p_8 = sum_5 = None
        sum_6 = q_8.sum(dim=2, keepdim=True)
        q_9 = q_8 / sum_6
        q_8 = sum_6 = None
        view_26 = p_9.view(1, 2, 1, 8, 8)
        p_9 = None
        expand_4 = view_26.expand(1, 2, 16, 8, 8)
        view_26 = None
        p_10 = expand_4.contiguous()
        expand_4 = None
        p_11 = p_10.view(1, 32, 8, 8)
        p_10 = None
        view_28 = q_9.view(1, 2, 1, 8, 8)
        q_9 = None
        expand_5 = view_28.expand(1, 2, 16, 8, 8)
        view_28 = None
        q_10 = expand_5.contiguous()
        expand_5 = None
        q_11 = q_10.view(1, 32, 8, 8)
        q_10 = None
        x_91 = p_11.view(32, -1, 1, 1)
        p_11 = None
        eye_4 = torch.eye(4, 4, dtype=torch.float32, device=device(type="cpu"))
        x_92 = x_91 * eye_4
        x_91 = eye_4 = None
        x_93 = x_92.view(32, 8, 8, 4, 4)
        x_92 = None
        split_8 = torch.functional.split(x_93, 1, dim=1)
        x_93 = None
        getitem_64 = split_8[0]
        getitem_65 = split_8[1]
        getitem_66 = split_8[2]
        getitem_67 = split_8[3]
        getitem_68 = split_8[4]
        getitem_69 = split_8[5]
        getitem_70 = split_8[6]
        getitem_71 = split_8[7]
        split_8 = None
        x_94 = torch.cat(
            (
                getitem_64,
                getitem_65,
                getitem_66,
                getitem_67,
                getitem_68,
                getitem_69,
                getitem_70,
                getitem_71,
            ),
            dim=3,
        )
        getitem_64 = (
            getitem_65
        ) = (
            getitem_66
        ) = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = None
        split_9 = torch.functional.split(x_94, 1, dim=2)
        x_94 = None
        getitem_72 = split_9[0]
        getitem_73 = split_9[1]
        getitem_74 = split_9[2]
        getitem_75 = split_9[3]
        getitem_76 = split_9[4]
        getitem_77 = split_9[5]
        getitem_78 = split_9[6]
        getitem_79 = split_9[7]
        split_9 = None
        x_95 = torch.cat(
            (
                getitem_72,
                getitem_73,
                getitem_74,
                getitem_75,
                getitem_76,
                getitem_77,
                getitem_78,
                getitem_79,
            ),
            dim=4,
        )
        getitem_72 = (
            getitem_73
        ) = (
            getitem_74
        ) = getitem_75 = getitem_76 = getitem_77 = getitem_78 = getitem_79 = None
        x_96 = x_95.view(1, 32, 32, 32)
        x_95 = None
        x_97 = q_11.view(32, -1, 1, 1)
        q_11 = None
        eye_5 = torch.eye(4, 4, dtype=torch.float32, device=device(type="cpu"))
        x_98 = x_97 * eye_5
        x_97 = eye_5 = None
        x_99 = x_98.view(32, 8, 8, 4, 4)
        x_98 = None
        split_10 = torch.functional.split(x_99, 1, dim=1)
        x_99 = None
        getitem_80 = split_10[0]
        getitem_81 = split_10[1]
        getitem_82 = split_10[2]
        getitem_83 = split_10[3]
        getitem_84 = split_10[4]
        getitem_85 = split_10[5]
        getitem_86 = split_10[6]
        getitem_87 = split_10[7]
        split_10 = None
        x_100 = torch.cat(
            (
                getitem_80,
                getitem_81,
                getitem_82,
                getitem_83,
                getitem_84,
                getitem_85,
                getitem_86,
                getitem_87,
            ),
            dim=3,
        )
        getitem_80 = (
            getitem_81
        ) = (
            getitem_82
        ) = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = None
        split_11 = torch.functional.split(x_100, 1, dim=2)
        x_100 = None
        getitem_88 = split_11[0]
        getitem_89 = split_11[1]
        getitem_90 = split_11[2]
        getitem_91 = split_11[3]
        getitem_92 = split_11[4]
        getitem_93 = split_11[5]
        getitem_94 = split_11[6]
        getitem_95 = split_11[7]
        split_11 = None
        x_101 = torch.cat(
            (
                getitem_88,
                getitem_89,
                getitem_90,
                getitem_91,
                getitem_92,
                getitem_93,
                getitem_94,
                getitem_95,
            ),
            dim=4,
        )
        getitem_88 = (
            getitem_89
        ) = (
            getitem_90
        ) = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = None
        x_102 = x_101.view(1, 32, 32, 32)
        x_101 = None
        y_6 = x_96.matmul(x_87)
        x_96 = x_87 = None
        y_7 = y_6.matmul(x_102)
        y_6 = x_102 = None
        x_103 = torch.conv2d(
            y_7,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        y_7 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_105 = torch.nn.functional.relu(x_104, inplace=True)
        x_104 = None
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_105 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_108 = torch.nn.functional.relu(x_107, inplace=True)
        x_107 = None
        y_8 = torch.nn.functional.dropout2d(x_108, 0.2, False, False)
        x_108 = None
        x_109 = y_8 + x_84
        y_8 = x_84 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_112 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_114 = x_111 + x_113
        x_111 = x_113 = None
        input_4 = torch.nn.functional.silu(x_114, inplace=True)
        x_114 = None
        x_115 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_117 = torch.nn.functional.silu(x_116, inplace=True)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_117 = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_120 = torch.nn.functional.silu(x_119, inplace=True)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_126 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        rp_3 = torch.nn.functional.adaptive_max_pool2d(x_126, (8, 1))
        cp_3 = torch.nn.functional.adaptive_max_pool2d(x_126, (1, 8))
        x_126 = None
        conv2d_36 = torch.conv2d(
            rp_3,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        rp_3 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = (None)
        view_36 = conv2d_36.view(1, 2, 8, 8)
        conv2d_36 = None
        p_12 = view_36.sigmoid()
        view_36 = None
        conv2d_37 = torch.conv2d(
            cp_3,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cp_3 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = (None)
        view_37 = conv2d_37.view(1, 2, 8, 8)
        conv2d_37 = None
        q_12 = view_37.sigmoid()
        view_37 = None
        sum_7 = p_12.sum(dim=3, keepdim=True)
        p_13 = p_12 / sum_7
        p_12 = sum_7 = None
        sum_8 = q_12.sum(dim=2, keepdim=True)
        q_13 = q_12 / sum_8
        q_12 = sum_8 = None
        view_38 = p_13.view(1, 2, 1, 8, 8)
        p_13 = None
        expand_6 = view_38.expand(1, 2, 16, 8, 8)
        view_38 = None
        p_14 = expand_6.contiguous()
        expand_6 = None
        p_15 = p_14.view(1, 32, 8, 8)
        p_14 = None
        view_40 = q_13.view(1, 2, 1, 8, 8)
        q_13 = None
        expand_7 = view_40.expand(1, 2, 16, 8, 8)
        view_40 = None
        q_14 = expand_7.contiguous()
        expand_7 = None
        q_15 = q_14.view(1, 32, 8, 8)
        q_14 = None
        x_127 = p_15.view(32, -1, 1, 1)
        p_15 = None
        eye_6 = torch.eye(4, 4, dtype=torch.float32, device=device(type="cpu"))
        x_128 = x_127 * eye_6
        x_127 = eye_6 = None
        x_129 = x_128.view(32, 8, 8, 4, 4)
        x_128 = None
        split_12 = torch.functional.split(x_129, 1, dim=1)
        x_129 = None
        getitem_96 = split_12[0]
        getitem_97 = split_12[1]
        getitem_98 = split_12[2]
        getitem_99 = split_12[3]
        getitem_100 = split_12[4]
        getitem_101 = split_12[5]
        getitem_102 = split_12[6]
        getitem_103 = split_12[7]
        split_12 = None
        x_130 = torch.cat(
            (
                getitem_96,
                getitem_97,
                getitem_98,
                getitem_99,
                getitem_100,
                getitem_101,
                getitem_102,
                getitem_103,
            ),
            dim=3,
        )
        getitem_96 = (
            getitem_97
        ) = (
            getitem_98
        ) = getitem_99 = getitem_100 = getitem_101 = getitem_102 = getitem_103 = None
        split_13 = torch.functional.split(x_130, 1, dim=2)
        x_130 = None
        getitem_104 = split_13[0]
        getitem_105 = split_13[1]
        getitem_106 = split_13[2]
        getitem_107 = split_13[3]
        getitem_108 = split_13[4]
        getitem_109 = split_13[5]
        getitem_110 = split_13[6]
        getitem_111 = split_13[7]
        split_13 = None
        x_131 = torch.cat(
            (
                getitem_104,
                getitem_105,
                getitem_106,
                getitem_107,
                getitem_108,
                getitem_109,
                getitem_110,
                getitem_111,
            ),
            dim=4,
        )
        getitem_104 = (
            getitem_105
        ) = (
            getitem_106
        ) = getitem_107 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = None
        x_132 = x_131.view(1, 32, 32, 32)
        x_131 = None
        x_133 = q_15.view(32, -1, 1, 1)
        q_15 = None
        eye_7 = torch.eye(4, 4, dtype=torch.float32, device=device(type="cpu"))
        x_134 = x_133 * eye_7
        x_133 = eye_7 = None
        x_135 = x_134.view(32, 8, 8, 4, 4)
        x_134 = None
        split_14 = torch.functional.split(x_135, 1, dim=1)
        x_135 = None
        getitem_112 = split_14[0]
        getitem_113 = split_14[1]
        getitem_114 = split_14[2]
        getitem_115 = split_14[3]
        getitem_116 = split_14[4]
        getitem_117 = split_14[5]
        getitem_118 = split_14[6]
        getitem_119 = split_14[7]
        split_14 = None
        x_136 = torch.cat(
            (
                getitem_112,
                getitem_113,
                getitem_114,
                getitem_115,
                getitem_116,
                getitem_117,
                getitem_118,
                getitem_119,
            ),
            dim=3,
        )
        getitem_112 = (
            getitem_113
        ) = (
            getitem_114
        ) = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = None
        split_15 = torch.functional.split(x_136, 1, dim=2)
        x_136 = None
        getitem_120 = split_15[0]
        getitem_121 = split_15[1]
        getitem_122 = split_15[2]
        getitem_123 = split_15[3]
        getitem_124 = split_15[4]
        getitem_125 = split_15[5]
        getitem_126 = split_15[6]
        getitem_127 = split_15[7]
        split_15 = None
        x_137 = torch.cat(
            (
                getitem_120,
                getitem_121,
                getitem_122,
                getitem_123,
                getitem_124,
                getitem_125,
                getitem_126,
                getitem_127,
            ),
            dim=4,
        )
        getitem_120 = (
            getitem_121
        ) = (
            getitem_122
        ) = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = None
        x_138 = x_137.view(1, 32, 32, 32)
        x_137 = None
        y_9 = x_132.matmul(x_123)
        x_132 = x_123 = None
        y_10 = y_9.matmul(x_138)
        y_9 = x_138 = None
        x_139 = torch.conv2d(
            y_10,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        y_10 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        y_11 = torch.nn.functional.dropout2d(x_144, 0.2, False, False)
        x_144 = None
        x_145 = y_11 + x_120
        y_11 = x_120 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_148 = x_147 + input_4
        x_147 = input_4 = None
        input_5 = torch.nn.functional.silu(x_148, inplace=True)
        x_148 = None
        x_149 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_151 = torch.nn.functional.silu(x_150, inplace=True)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        x_151 = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_154 = torch.nn.functional.silu(x_153, inplace=True)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_160 = torch.nn.functional.relu(x_159, inplace=True)
        x_159 = None
        rp_4 = torch.nn.functional.adaptive_max_pool2d(x_160, (8, 1))
        cp_4 = torch.nn.functional.adaptive_max_pool2d(x_160, (1, 8))
        x_160 = None
        conv2d_45 = torch.conv2d(
            rp_4,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        rp_4 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = (None)
        view_48 = conv2d_45.view(1, 2, 8, 8)
        conv2d_45 = None
        p_16 = view_48.sigmoid()
        view_48 = None
        conv2d_46 = torch.conv2d(
            cp_4,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cp_4 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = (None)
        view_49 = conv2d_46.view(1, 2, 8, 8)
        conv2d_46 = None
        q_16 = view_49.sigmoid()
        view_49 = None
        sum_9 = p_16.sum(dim=3, keepdim=True)
        p_17 = p_16 / sum_9
        p_16 = sum_9 = None
        sum_10 = q_16.sum(dim=2, keepdim=True)
        q_17 = q_16 / sum_10
        q_16 = sum_10 = None
        view_50 = p_17.view(1, 2, 1, 8, 8)
        p_17 = None
        expand_8 = view_50.expand(1, 2, 32, 8, 8)
        view_50 = None
        p_18 = expand_8.contiguous()
        expand_8 = None
        p_19 = p_18.view(1, 64, 8, 8)
        p_18 = None
        view_52 = q_17.view(1, 2, 1, 8, 8)
        q_17 = None
        expand_9 = view_52.expand(1, 2, 32, 8, 8)
        view_52 = None
        q_18 = expand_9.contiguous()
        expand_9 = None
        q_19 = q_18.view(1, 64, 8, 8)
        q_18 = None
        x_161 = p_19.view(64, -1, 1, 1)
        p_19 = None
        eye_8 = torch.eye(2, 2, dtype=torch.float32, device=device(type="cpu"))
        x_162 = x_161 * eye_8
        x_161 = eye_8 = None
        x_163 = x_162.view(64, 8, 8, 2, 2)
        x_162 = None
        split_16 = torch.functional.split(x_163, 1, dim=1)
        x_163 = None
        getitem_128 = split_16[0]
        getitem_129 = split_16[1]
        getitem_130 = split_16[2]
        getitem_131 = split_16[3]
        getitem_132 = split_16[4]
        getitem_133 = split_16[5]
        getitem_134 = split_16[6]
        getitem_135 = split_16[7]
        split_16 = None
        x_164 = torch.cat(
            (
                getitem_128,
                getitem_129,
                getitem_130,
                getitem_131,
                getitem_132,
                getitem_133,
                getitem_134,
                getitem_135,
            ),
            dim=3,
        )
        getitem_128 = (
            getitem_129
        ) = (
            getitem_130
        ) = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = None
        split_17 = torch.functional.split(x_164, 1, dim=2)
        x_164 = None
        getitem_136 = split_17[0]
        getitem_137 = split_17[1]
        getitem_138 = split_17[2]
        getitem_139 = split_17[3]
        getitem_140 = split_17[4]
        getitem_141 = split_17[5]
        getitem_142 = split_17[6]
        getitem_143 = split_17[7]
        split_17 = None
        x_165 = torch.cat(
            (
                getitem_136,
                getitem_137,
                getitem_138,
                getitem_139,
                getitem_140,
                getitem_141,
                getitem_142,
                getitem_143,
            ),
            dim=4,
        )
        getitem_136 = (
            getitem_137
        ) = (
            getitem_138
        ) = getitem_139 = getitem_140 = getitem_141 = getitem_142 = getitem_143 = None
        x_166 = x_165.view(1, 64, 16, 16)
        x_165 = None
        x_167 = q_19.view(64, -1, 1, 1)
        q_19 = None
        eye_9 = torch.eye(2, 2, dtype=torch.float32, device=device(type="cpu"))
        x_168 = x_167 * eye_9
        x_167 = eye_9 = None
        x_169 = x_168.view(64, 8, 8, 2, 2)
        x_168 = None
        split_18 = torch.functional.split(x_169, 1, dim=1)
        x_169 = None
        getitem_144 = split_18[0]
        getitem_145 = split_18[1]
        getitem_146 = split_18[2]
        getitem_147 = split_18[3]
        getitem_148 = split_18[4]
        getitem_149 = split_18[5]
        getitem_150 = split_18[6]
        getitem_151 = split_18[7]
        split_18 = None
        x_170 = torch.cat(
            (
                getitem_144,
                getitem_145,
                getitem_146,
                getitem_147,
                getitem_148,
                getitem_149,
                getitem_150,
                getitem_151,
            ),
            dim=3,
        )
        getitem_144 = (
            getitem_145
        ) = (
            getitem_146
        ) = getitem_147 = getitem_148 = getitem_149 = getitem_150 = getitem_151 = None
        split_19 = torch.functional.split(x_170, 1, dim=2)
        x_170 = None
        getitem_152 = split_19[0]
        getitem_153 = split_19[1]
        getitem_154 = split_19[2]
        getitem_155 = split_19[3]
        getitem_156 = split_19[4]
        getitem_157 = split_19[5]
        getitem_158 = split_19[6]
        getitem_159 = split_19[7]
        split_19 = None
        x_171 = torch.cat(
            (
                getitem_152,
                getitem_153,
                getitem_154,
                getitem_155,
                getitem_156,
                getitem_157,
                getitem_158,
                getitem_159,
            ),
            dim=4,
        )
        getitem_152 = (
            getitem_153
        ) = (
            getitem_154
        ) = getitem_155 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = None
        x_172 = x_171.view(1, 64, 16, 16)
        x_171 = None
        y_12 = x_166.matmul(x_157)
        x_166 = x_157 = None
        y_13 = y_12.matmul(x_172)
        y_12 = x_172 = None
        x_173 = torch.conv2d(
            y_13,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        y_13 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_175 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_178 = torch.nn.functional.relu(x_177, inplace=True)
        x_177 = None
        y_14 = torch.nn.functional.dropout2d(x_178, 0.2, False, False)
        x_178 = None
        x_179 = y_14 + x_154
        y_14 = x_154 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_182 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_184 = x_181 + x_183
        x_181 = x_183 = None
        input_6 = torch.nn.functional.silu(x_184, inplace=True)
        x_184 = None
        x_185 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_186 = torch.nn.functional.batch_norm(
            x_185,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_185 = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_187 = torch.nn.functional.silu(x_186, inplace=True)
        x_186 = None
        x_188 = torch.conv2d(
            x_187,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_187 = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_189 = torch.nn.functional.batch_norm(
            x_188,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_188 = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_190 = torch.nn.functional.silu(x_189, inplace=True)
        x_189 = None
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_191 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_196 = torch.nn.functional.relu(x_195, inplace=True)
        x_195 = None
        rp_5 = torch.nn.functional.adaptive_max_pool2d(x_196, (8, 1))
        cp_5 = torch.nn.functional.adaptive_max_pool2d(x_196, (1, 8))
        x_196 = None
        conv2d_55 = torch.conv2d(
            rp_5,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        rp_5 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = (None)
        view_60 = conv2d_55.view(1, 2, 8, 8)
        conv2d_55 = None
        p_20 = view_60.sigmoid()
        view_60 = None
        conv2d_56 = torch.conv2d(
            cp_5,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cp_5 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = (None)
        view_61 = conv2d_56.view(1, 2, 8, 8)
        conv2d_56 = None
        q_20 = view_61.sigmoid()
        view_61 = None
        sum_11 = p_20.sum(dim=3, keepdim=True)
        p_21 = p_20 / sum_11
        p_20 = sum_11 = None
        sum_12 = q_20.sum(dim=2, keepdim=True)
        q_21 = q_20 / sum_12
        q_20 = sum_12 = None
        view_62 = p_21.view(1, 2, 1, 8, 8)
        p_21 = None
        expand_10 = view_62.expand(1, 2, 32, 8, 8)
        view_62 = None
        p_22 = expand_10.contiguous()
        expand_10 = None
        p_23 = p_22.view(1, 64, 8, 8)
        p_22 = None
        view_64 = q_21.view(1, 2, 1, 8, 8)
        q_21 = None
        expand_11 = view_64.expand(1, 2, 32, 8, 8)
        view_64 = None
        q_22 = expand_11.contiguous()
        expand_11 = None
        q_23 = q_22.view(1, 64, 8, 8)
        q_22 = None
        x_197 = p_23.view(64, -1, 1, 1)
        p_23 = None
        eye_10 = torch.eye(2, 2, dtype=torch.float32, device=device(type="cpu"))
        x_198 = x_197 * eye_10
        x_197 = eye_10 = None
        x_199 = x_198.view(64, 8, 8, 2, 2)
        x_198 = None
        split_20 = torch.functional.split(x_199, 1, dim=1)
        x_199 = None
        getitem_160 = split_20[0]
        getitem_161 = split_20[1]
        getitem_162 = split_20[2]
        getitem_163 = split_20[3]
        getitem_164 = split_20[4]
        getitem_165 = split_20[5]
        getitem_166 = split_20[6]
        getitem_167 = split_20[7]
        split_20 = None
        x_200 = torch.cat(
            (
                getitem_160,
                getitem_161,
                getitem_162,
                getitem_163,
                getitem_164,
                getitem_165,
                getitem_166,
                getitem_167,
            ),
            dim=3,
        )
        getitem_160 = (
            getitem_161
        ) = (
            getitem_162
        ) = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_167 = None
        split_21 = torch.functional.split(x_200, 1, dim=2)
        x_200 = None
        getitem_168 = split_21[0]
        getitem_169 = split_21[1]
        getitem_170 = split_21[2]
        getitem_171 = split_21[3]
        getitem_172 = split_21[4]
        getitem_173 = split_21[5]
        getitem_174 = split_21[6]
        getitem_175 = split_21[7]
        split_21 = None
        x_201 = torch.cat(
            (
                getitem_168,
                getitem_169,
                getitem_170,
                getitem_171,
                getitem_172,
                getitem_173,
                getitem_174,
                getitem_175,
            ),
            dim=4,
        )
        getitem_168 = (
            getitem_169
        ) = (
            getitem_170
        ) = getitem_171 = getitem_172 = getitem_173 = getitem_174 = getitem_175 = None
        x_202 = x_201.view(1, 64, 16, 16)
        x_201 = None
        x_203 = q_23.view(64, -1, 1, 1)
        q_23 = None
        eye_11 = torch.eye(2, 2, dtype=torch.float32, device=device(type="cpu"))
        x_204 = x_203 * eye_11
        x_203 = eye_11 = None
        x_205 = x_204.view(64, 8, 8, 2, 2)
        x_204 = None
        split_22 = torch.functional.split(x_205, 1, dim=1)
        x_205 = None
        getitem_176 = split_22[0]
        getitem_177 = split_22[1]
        getitem_178 = split_22[2]
        getitem_179 = split_22[3]
        getitem_180 = split_22[4]
        getitem_181 = split_22[5]
        getitem_182 = split_22[6]
        getitem_183 = split_22[7]
        split_22 = None
        x_206 = torch.cat(
            (
                getitem_176,
                getitem_177,
                getitem_178,
                getitem_179,
                getitem_180,
                getitem_181,
                getitem_182,
                getitem_183,
            ),
            dim=3,
        )
        getitem_176 = (
            getitem_177
        ) = (
            getitem_178
        ) = getitem_179 = getitem_180 = getitem_181 = getitem_182 = getitem_183 = None
        split_23 = torch.functional.split(x_206, 1, dim=2)
        x_206 = None
        getitem_184 = split_23[0]
        getitem_185 = split_23[1]
        getitem_186 = split_23[2]
        getitem_187 = split_23[3]
        getitem_188 = split_23[4]
        getitem_189 = split_23[5]
        getitem_190 = split_23[6]
        getitem_191 = split_23[7]
        split_23 = None
        x_207 = torch.cat(
            (
                getitem_184,
                getitem_185,
                getitem_186,
                getitem_187,
                getitem_188,
                getitem_189,
                getitem_190,
                getitem_191,
            ),
            dim=4,
        )
        getitem_184 = (
            getitem_185
        ) = (
            getitem_186
        ) = getitem_187 = getitem_188 = getitem_189 = getitem_190 = getitem_191 = None
        x_208 = x_207.view(1, 64, 16, 16)
        x_207 = None
        y_15 = x_202.matmul(x_193)
        x_202 = x_193 = None
        y_16 = y_15.matmul(x_208)
        y_15 = x_208 = None
        x_209 = torch.conv2d(
            y_16,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        y_16 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_210 = torch.nn.functional.batch_norm(
            x_209,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_209 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_211 = torch.nn.functional.relu(x_210, inplace=True)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_211 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_214 = torch.nn.functional.relu(x_213, inplace=True)
        x_213 = None
        y_17 = torch.nn.functional.dropout2d(x_214, 0.2, False, False)
        x_214 = None
        x_215 = y_17 + x_190
        y_17 = x_190 = None
        x_216 = torch.conv2d(
            x_215,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_215 = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_216 = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_218 = x_217 + input_6
        x_217 = input_6 = None
        input_7 = torch.nn.functional.silu(x_218, inplace=True)
        x_218 = None
        x_219 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_220 = torch.nn.functional.batch_norm(
            x_219,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_219 = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_221 = torch.nn.functional.silu(x_220, inplace=True)
        x_220 = None
        x_222 = torch.conv2d(
            x_221,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            16,
        )
        x_221 = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_223 = torch.nn.functional.batch_norm(
            x_222,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_222 = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_224 = torch.nn.functional.silu(x_223, inplace=True)
        x_223 = None
        x_225 = torch.conv2d(
            x_224,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_226 = torch.nn.functional.batch_norm(
            x_225,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_225 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_227 = torch.nn.functional.relu(x_226, inplace=True)
        x_226 = None
        x_228 = torch.conv2d(
            x_227,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_229 = torch.nn.functional.batch_norm(
            x_228,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_228 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_230 = torch.nn.functional.relu(x_229, inplace=True)
        x_229 = None
        rp_6 = torch.nn.functional.adaptive_max_pool2d(x_230, (8, 1))
        cp_6 = torch.nn.functional.adaptive_max_pool2d(x_230, (1, 8))
        x_230 = None
        conv2d_64 = torch.conv2d(
            rp_6,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        rp_6 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = (None)
        view_72 = conv2d_64.view(1, 2, 8, 8)
        conv2d_64 = None
        p_24 = view_72.sigmoid()
        view_72 = None
        conv2d_65 = torch.conv2d(
            cp_6,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cp_6 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = (None)
        view_73 = conv2d_65.view(1, 2, 8, 8)
        conv2d_65 = None
        q_24 = view_73.sigmoid()
        view_73 = None
        sum_13 = p_24.sum(dim=3, keepdim=True)
        p_25 = p_24 / sum_13
        p_24 = sum_13 = None
        sum_14 = q_24.sum(dim=2, keepdim=True)
        q_25 = q_24 / sum_14
        q_24 = sum_14 = None
        view_74 = p_25.view(1, 2, 1, 8, 8)
        p_25 = None
        expand_12 = view_74.expand(1, 2, 64, 8, 8)
        view_74 = None
        p_26 = expand_12.contiguous()
        expand_12 = None
        p_27 = p_26.view(1, 128, 8, 8)
        p_26 = None
        view_76 = q_25.view(1, 2, 1, 8, 8)
        q_25 = None
        expand_13 = view_76.expand(1, 2, 64, 8, 8)
        view_76 = None
        q_26 = expand_13.contiguous()
        expand_13 = None
        q_27 = q_26.view(1, 128, 8, 8)
        q_26 = None
        y_18 = p_27.matmul(x_227)
        p_27 = x_227 = None
        y_19 = y_18.matmul(q_27)
        y_18 = q_27 = None
        x_231 = torch.conv2d(
            y_19,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        y_19 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_232 = torch.nn.functional.batch_norm(
            x_231,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_231 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_233 = torch.nn.functional.relu(x_232, inplace=True)
        x_232 = None
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_233 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_attn_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_236 = torch.nn.functional.relu(x_235, inplace=True)
        x_235 = None
        y_20 = torch.nn.functional.dropout2d(x_236, 0.2, False, False)
        x_236 = None
        x_237 = y_20 + x_224
        y_20 = x_224 = None
        x_238 = torch.conv2d(
            x_237,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_237 = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_239 = torch.nn.functional.batch_norm(
            x_238,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_238 = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_240 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_241 = torch.nn.functional.batch_norm(
            x_240,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_240 = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_242 = x_239 + x_241
        x_239 = x_241 = None
        input_8 = torch.nn.functional.silu(x_242, inplace=True)
        x_242 = None
        x_243 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_244 = torch.nn.functional.batch_norm(
            x_243,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_243 = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_ = (None)
        x_245 = torch.nn.functional.silu(x_244, inplace=True)
        x_244 = None
        x_246 = torch.conv2d(
            x_245,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            16,
        )
        x_245 = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_ = (None)
        x_247 = torch.nn.functional.batch_norm(
            x_246,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_246 = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_ = (None)
        x_248 = torch.nn.functional.silu(x_247, inplace=True)
        x_247 = None
        x_249 = torch.conv2d(
            x_248,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_250 = torch.nn.functional.batch_norm(
            x_249,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_249 = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_251 = torch.nn.functional.relu(x_250, inplace=True)
        x_250 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_253 = torch.nn.functional.batch_norm(
            x_252,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_252 = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_254 = torch.nn.functional.relu(x_253, inplace=True)
        x_253 = None
        rp_7 = torch.nn.functional.adaptive_max_pool2d(x_254, (8, 1))
        cp_7 = torch.nn.functional.adaptive_max_pool2d(x_254, (1, 8))
        x_254 = None
        conv2d_74 = torch.conv2d(
            rp_7,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        rp_7 = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_p_parameters_bias_ = (None)
        view_78 = conv2d_74.view(1, 2, 8, 8)
        conv2d_74 = None
        p_28 = view_78.sigmoid()
        view_78 = None
        conv2d_75 = torch.conv2d(
            cp_7,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cp_7 = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv_q_parameters_bias_ = (None)
        view_79 = conv2d_75.view(1, 2, 8, 8)
        conv2d_75 = None
        q_28 = view_79.sigmoid()
        view_79 = None
        sum_15 = p_28.sum(dim=3, keepdim=True)
        p_29 = p_28 / sum_15
        p_28 = sum_15 = None
        sum_16 = q_28.sum(dim=2, keepdim=True)
        q_29 = q_28 / sum_16
        q_28 = sum_16 = None
        view_80 = p_29.view(1, 2, 1, 8, 8)
        p_29 = None
        expand_14 = view_80.expand(1, 2, 64, 8, 8)
        view_80 = None
        p_30 = expand_14.contiguous()
        expand_14 = None
        p_31 = p_30.view(1, 128, 8, 8)
        p_30 = None
        view_82 = q_29.view(1, 2, 1, 8, 8)
        q_29 = None
        expand_15 = view_82.expand(1, 2, 64, 8, 8)
        view_82 = None
        q_30 = expand_15.contiguous()
        expand_15 = None
        q_31 = q_30.view(1, 128, 8, 8)
        q_30 = None
        y_21 = p_31.matmul(x_251)
        p_31 = x_251 = None
        y_22 = y_21.matmul(q_31)
        y_21 = q_31 = None
        x_255 = torch.conv2d(
            y_22,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        y_22 = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_256 = torch.nn.functional.batch_norm(
            x_255,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_255 = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_ba_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_257 = torch.nn.functional.relu(x_256, inplace=True)
        x_256 = None
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_259 = torch.nn.functional.batch_norm(
            x_258,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_258 = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_attn_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_260 = torch.nn.functional.relu(x_259, inplace=True)
        x_259 = None
        y_23 = torch.nn.functional.dropout2d(x_260, 0.2, False, False)
        x_260 = None
        x_261 = y_23 + x_248
        y_23 = x_248 = None
        x_262 = torch.conv2d(
            x_261,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_261 = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_ = (None)
        x_263 = torch.nn.functional.batch_norm(
            x_262,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_262 = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_ = (None)
        x_264 = x_263 + input_8
        x_263 = input_8 = None
        input_9 = torch.nn.functional.silu(x_264, inplace=True)
        x_264 = None
        x_265 = torch.nn.functional.adaptive_avg_pool2d(input_9, 1)
        input_9 = None
        x_266 = x_265.flatten(1, -1)
        x_265 = None
        x_267 = torch.nn.functional.dropout(x_266, 0.0, False, False)
        x_266 = None
        x_268 = torch._C._nn.linear(
            x_267,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_267 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_268,)
