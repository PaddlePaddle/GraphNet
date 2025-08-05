import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_conv1_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_conv1_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv1_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv1_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv1_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_features_modules_conv1_1_modules_conv_parameters_weight_ = (
            L_self_modules_features_modules_conv1_1_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_features_modules_conv1_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_features_modules_conv1_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_features_modules_conv1_1_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_conv1_1_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_conv1_1_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_conv1_1_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_conv1_1_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_conv1_1_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_var_ = L_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_var_
        l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_weight_ = L_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_weight_
        l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_bias_ = L_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_bias_
        l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_conv_parameters_weight_ = L_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_conv_parameters_weight_
        l_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_mean_ = (
            L_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_mean_
        )
        l_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_bias_
        )
        l_self_modules_classifier_parameters_weight_ = (
            L_self_modules_classifier_parameters_weight_
        )
        l_self_modules_classifier_parameters_bias_ = (
            L_self_modules_classifier_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_features_modules_conv1_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_features_modules_conv1_1_modules_conv_parameters_weight_
        ) = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_features_modules_conv1_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv1_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv1_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv1_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x = (
            l_self_modules_features_modules_conv1_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_conv1_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_conv1_1_modules_bn_parameters_weight_
        ) = l_self_modules_features_modules_conv1_1_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        input_1 = torch.nn.functional.max_pool2d(
            x_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_2 = None
        x_3 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_bn_parameters_bias_ = (None)
        x_4 = torch.nn.functional.relu(x_3, inplace=True)
        x_3 = None
        x_s = torch.conv2d(
            x_4,
            l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_features_modules_conv2_1_modules_c1x1_w_s1_modules_conv_parameters_weight_ = (None)
        x_s1 = x_s[
            (
                slice(None, None, None),
                slice(None, 256, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s2 = x_s[
            (
                slice(None, None, None),
                slice(256, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s = None
        x_5 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_1 = l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_6 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        x_in = torch.conv2d(
            x_6,
            l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_features_modules_conv2_1_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_in,
            l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in = l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_in_1 = torch.conv2d(
            x_8,
            l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_8 = l_self_modules_features_modules_conv2_1_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_9 = torch.nn.functional.batch_norm(
            x_in_1,
            l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_1 = l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_10 = torch.nn.functional.relu(x_9, inplace=True)
        x_9 = None
        x_in_2 = torch.conv2d(
            x_10,
            l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_features_modules_conv2_1_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1 = x_in_2[
            (
                slice(None, None, None),
                slice(None, 256, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2 = x_in_2[
            (
                slice(None, None, None),
                slice(256, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_2 = None
        resid = x_s1 + out1
        x_s1 = out1 = None
        dense = torch.cat([x_s2, out2], dim=1)
        x_s2 = out2 = None
        x_in_3 = torch.cat((resid, dense), dim=1)
        x_11 = torch.nn.functional.batch_norm(
            x_in_3,
            l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_3 = l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        x_in_4 = torch.conv2d(
            x_12,
            l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_in_4,
            l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_4 = l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        x_in_5 = torch.conv2d(
            x_14,
            l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_14 = l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_in_5,
            l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_5 = l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_16 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        x_in_6 = torch.conv2d(
            x_16,
            l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_1 = x_in_6[
            (
                slice(None, None, None),
                slice(None, 256, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_1 = x_in_6[
            (
                slice(None, None, None),
                slice(256, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_6 = None
        resid_1 = resid + out1_1
        resid = out1_1 = None
        dense_1 = torch.cat([dense, out2_1], dim=1)
        dense = out2_1 = None
        x_in_7 = torch.cat((resid_1, dense_1), dim=1)
        x_17 = torch.nn.functional.batch_norm(
            x_in_7,
            l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_7 = l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        x_in_8 = torch.conv2d(
            x_18,
            l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_19 = torch.nn.functional.batch_norm(
            x_in_8,
            l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_8 = l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_in_9 = torch.conv2d(
            x_20,
            l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_20 = l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_21 = torch.nn.functional.batch_norm(
            x_in_9,
            l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_9 = l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_in_10 = torch.conv2d(
            x_22,
            l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_2 = x_in_10[
            (
                slice(None, None, None),
                slice(None, 256, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_2 = x_in_10[
            (
                slice(None, None, None),
                slice(256, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_10 = None
        resid_2 = resid_1 + out1_2
        resid_1 = out1_2 = None
        dense_2 = torch.cat([dense_1, out2_2], dim=1)
        dense_1 = out2_2 = None
        x_in_11 = torch.cat((resid_2, dense_2), dim=1)
        x_23 = torch.nn.functional.batch_norm(
            x_in_11,
            l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_11 = l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        x_in_12 = torch.conv2d(
            x_24,
            l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_24 = l_self_modules_features_modules_conv2_4_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_in_12,
            l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_12 = l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_26 = torch.nn.functional.relu(x_25, inplace=True)
        x_25 = None
        x_in_13 = torch.conv2d(
            x_26,
            l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_26 = l_self_modules_features_modules_conv2_4_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_in_13,
            l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_13 = l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.relu(x_27, inplace=True)
        x_27 = None
        x_in_14 = torch.conv2d(
            x_28,
            l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_features_modules_conv2_4_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_3 = x_in_14[
            (
                slice(None, None, None),
                slice(None, 256, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_3 = x_in_14[
            (
                slice(None, None, None),
                slice(256, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_14 = None
        resid_3 = resid_2 + out1_3
        resid_2 = out1_3 = None
        dense_3 = torch.cat([dense_2, out2_3], dim=1)
        dense_2 = out2_3 = None
        x_in_15 = torch.cat((resid_3, dense_3), dim=1)
        resid_3 = dense_3 = None
        x_29 = torch.nn.functional.batch_norm(
            x_in_15,
            l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_bias_ = (None)
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        x_s_1 = torch.conv2d(
            x_30,
            l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_conv_parameters_weight_ = (None)
        x_s1_1 = x_s_1[
            (
                slice(None, None, None),
                slice(None, 512, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s2_1 = x_s_1[
            (
                slice(None, None, None),
                slice(512, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s_1 = None
        x_31 = torch.nn.functional.batch_norm(
            x_in_15,
            l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_15 = l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x_in_16 = torch.conv2d(
            x_32,
            l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_in_16,
            l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_16 = l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        x_in_17 = torch.conv2d(
            x_34,
            l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            50,
        )
        x_34 = l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_in_17,
            l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_17 = l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_36 = torch.nn.functional.relu(x_35, inplace=True)
        x_35 = None
        x_in_18 = torch.conv2d(
            x_36,
            l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_36 = l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_4 = x_in_18[
            (
                slice(None, None, None),
                slice(None, 512, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_4 = x_in_18[
            (
                slice(None, None, None),
                slice(512, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_18 = None
        resid_4 = x_s1_1 + out1_4
        x_s1_1 = out1_4 = None
        dense_4 = torch.cat([x_s2_1, out2_4], dim=1)
        x_s2_1 = out2_4 = None
        x_in_19 = torch.cat((resid_4, dense_4), dim=1)
        x_37 = torch.nn.functional.batch_norm(
            x_in_19,
            l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_19 = l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_in_20 = torch.conv2d(
            x_38,
            l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_in_20,
            l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_20 = l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_in_21 = torch.conv2d(
            x_40,
            l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_40 = l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_in_21,
            l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_21 = l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_in_22 = torch.conv2d(
            x_42,
            l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_5 = x_in_22[
            (
                slice(None, None, None),
                slice(None, 512, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_5 = x_in_22[
            (
                slice(None, None, None),
                slice(512, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_22 = None
        resid_5 = resid_4 + out1_5
        resid_4 = out1_5 = None
        dense_5 = torch.cat([dense_4, out2_5], dim=1)
        dense_4 = out2_5 = None
        x_in_23 = torch.cat((resid_5, dense_5), dim=1)
        x_43 = torch.nn.functional.batch_norm(
            x_in_23,
            l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_23 = l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_44 = torch.nn.functional.relu(x_43, inplace=True)
        x_43 = None
        x_in_24 = torch.conv2d(
            x_44,
            l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_in_24,
            l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_24 = l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_in_25 = torch.conv2d(
            x_46,
            l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_46 = l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_in_25,
            l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_25 = l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_48 = torch.nn.functional.relu(x_47, inplace=True)
        x_47 = None
        x_in_26 = torch.conv2d(
            x_48,
            l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_6 = x_in_26[
            (
                slice(None, None, None),
                slice(None, 512, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_6 = x_in_26[
            (
                slice(None, None, None),
                slice(512, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_26 = None
        resid_6 = resid_5 + out1_6
        resid_5 = out1_6 = None
        dense_6 = torch.cat([dense_5, out2_6], dim=1)
        dense_5 = out2_6 = None
        x_in_27 = torch.cat((resid_6, dense_6), dim=1)
        x_49 = torch.nn.functional.batch_norm(
            x_in_27,
            l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_27 = l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_50 = torch.nn.functional.relu(x_49, inplace=True)
        x_49 = None
        x_in_28 = torch.conv2d(
            x_50,
            l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_50 = l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_in_28,
            l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_28 = l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_in_29 = torch.conv2d(
            x_52,
            l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_52 = l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_in_29,
            l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_29 = l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_54 = torch.nn.functional.relu(x_53, inplace=True)
        x_53 = None
        x_in_30 = torch.conv2d(
            x_54,
            l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_7 = x_in_30[
            (
                slice(None, None, None),
                slice(None, 512, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_7 = x_in_30[
            (
                slice(None, None, None),
                slice(512, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_30 = None
        resid_7 = resid_6 + out1_7
        resid_6 = out1_7 = None
        dense_7 = torch.cat([dense_6, out2_7], dim=1)
        dense_6 = out2_7 = None
        x_in_31 = torch.cat((resid_7, dense_7), dim=1)
        x_55 = torch.nn.functional.batch_norm(
            x_in_31,
            l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_31 = l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        x_in_32 = torch.conv2d(
            x_56,
            l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_56 = l_self_modules_features_modules_conv3_5_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_in_32,
            l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_32 = l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        x_in_33 = torch.conv2d(
            x_58,
            l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_58 = l_self_modules_features_modules_conv3_5_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_in_33,
            l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_33 = l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_in_34 = torch.conv2d(
            x_60,
            l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_features_modules_conv3_5_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_8 = x_in_34[
            (
                slice(None, None, None),
                slice(None, 512, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_8 = x_in_34[
            (
                slice(None, None, None),
                slice(512, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_34 = None
        resid_8 = resid_7 + out1_8
        resid_7 = out1_8 = None
        dense_8 = torch.cat([dense_7, out2_8], dim=1)
        dense_7 = out2_8 = None
        x_in_35 = torch.cat((resid_8, dense_8), dim=1)
        x_61 = torch.nn.functional.batch_norm(
            x_in_35,
            l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_35 = l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_62 = torch.nn.functional.relu(x_61, inplace=True)
        x_61 = None
        x_in_36 = torch.conv2d(
            x_62,
            l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_features_modules_conv3_6_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
            x_in_36,
            l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_36 = l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        x_in_37 = torch.conv2d(
            x_64,
            l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_64 = l_self_modules_features_modules_conv3_6_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_in_37,
            l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_37 = l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_66 = torch.nn.functional.relu(x_65, inplace=True)
        x_65 = None
        x_in_38 = torch.conv2d(
            x_66,
            l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_66 = l_self_modules_features_modules_conv3_6_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_9 = x_in_38[
            (
                slice(None, None, None),
                slice(None, 512, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_9 = x_in_38[
            (
                slice(None, None, None),
                slice(512, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_38 = None
        resid_9 = resid_8 + out1_9
        resid_8 = out1_9 = None
        dense_9 = torch.cat([dense_8, out2_9], dim=1)
        dense_8 = out2_9 = None
        x_in_39 = torch.cat((resid_9, dense_9), dim=1)
        x_67 = torch.nn.functional.batch_norm(
            x_in_39,
            l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_39 = l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        x_in_40 = torch.conv2d(
            x_68,
            l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_features_modules_conv3_7_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_in_40,
            l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_40 = l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_70 = torch.nn.functional.relu(x_69, inplace=True)
        x_69 = None
        x_in_41 = torch.conv2d(
            x_70,
            l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_70 = l_self_modules_features_modules_conv3_7_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_in_41,
            l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_41 = l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_in_42 = torch.conv2d(
            x_72,
            l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_features_modules_conv3_7_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_10 = x_in_42[
            (
                slice(None, None, None),
                slice(None, 512, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_10 = x_in_42[
            (
                slice(None, None, None),
                slice(512, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_42 = None
        resid_10 = resid_9 + out1_10
        resid_9 = out1_10 = None
        dense_10 = torch.cat([dense_9, out2_10], dim=1)
        dense_9 = out2_10 = None
        x_in_43 = torch.cat((resid_10, dense_10), dim=1)
        x_73 = torch.nn.functional.batch_norm(
            x_in_43,
            l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_43 = l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        x_in_44 = torch.conv2d(
            x_74,
            l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_features_modules_conv3_8_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_75 = torch.nn.functional.batch_norm(
            x_in_44,
            l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_44 = l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_76 = torch.nn.functional.relu(x_75, inplace=True)
        x_75 = None
        x_in_45 = torch.conv2d(
            x_76,
            l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_76 = l_self_modules_features_modules_conv3_8_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_in_45,
            l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_45 = l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        x_in_46 = torch.conv2d(
            x_78,
            l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_features_modules_conv3_8_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_11 = x_in_46[
            (
                slice(None, None, None),
                slice(None, 512, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_11 = x_in_46[
            (
                slice(None, None, None),
                slice(512, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_46 = None
        resid_11 = resid_10 + out1_11
        resid_10 = out1_11 = None
        dense_11 = torch.cat([dense_10, out2_11], dim=1)
        dense_10 = out2_11 = None
        x_in_47 = torch.cat((resid_11, dense_11), dim=1)
        resid_11 = dense_11 = None
        x_79 = torch.nn.functional.batch_norm(
            x_in_47,
            l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_bias_ = (None)
        x_80 = torch.nn.functional.relu(x_79, inplace=True)
        x_79 = None
        x_s_2 = torch.conv2d(
            x_80,
            l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_80 = l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_conv_parameters_weight_ = (None)
        x_s1_2 = x_s_2[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s2_2 = x_s_2[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s_2 = None
        x_81 = torch.nn.functional.batch_norm(
            x_in_47,
            l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_47 = l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_in_48 = torch.conv2d(
            x_82,
            l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_in_48,
            l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_48 = l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        x_in_49 = torch.conv2d(
            x_84,
            l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            50,
        )
        x_84 = l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_85 = torch.nn.functional.batch_norm(
            x_in_49,
            l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_49 = l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        x_in_50 = torch.conv2d(
            x_86,
            l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_12 = x_in_50[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_12 = x_in_50[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_50 = None
        resid_12 = x_s1_2 + out1_12
        x_s1_2 = out1_12 = None
        dense_12 = torch.cat([x_s2_2, out2_12], dim=1)
        x_s2_2 = out2_12 = None
        x_in_51 = torch.cat((resid_12, dense_12), dim=1)
        x_87 = torch.nn.functional.batch_norm(
            x_in_51,
            l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_51 = l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_88 = torch.nn.functional.relu(x_87, inplace=True)
        x_87 = None
        x_in_52 = torch.conv2d(
            x_88,
            l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_in_52,
            l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_52 = l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_in_53 = torch.conv2d(
            x_90,
            l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_90 = l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_91 = torch.nn.functional.batch_norm(
            x_in_53,
            l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_53 = l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_92 = torch.nn.functional.relu(x_91, inplace=True)
        x_91 = None
        x_in_54 = torch.conv2d(
            x_92,
            l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_92 = l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_13 = x_in_54[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_13 = x_in_54[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_54 = None
        resid_13 = resid_12 + out1_13
        resid_12 = out1_13 = None
        dense_13 = torch.cat([dense_12, out2_13], dim=1)
        dense_12 = out2_13 = None
        x_in_55 = torch.cat((resid_13, dense_13), dim=1)
        x_93 = torch.nn.functional.batch_norm(
            x_in_55,
            l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_55 = l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        x_in_56 = torch.conv2d(
            x_94,
            l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_in_56,
            l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_56 = l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        x_in_57 = torch.conv2d(
            x_96,
            l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_96 = l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_97 = torch.nn.functional.batch_norm(
            x_in_57,
            l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_57 = l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_98 = torch.nn.functional.relu(x_97, inplace=True)
        x_97 = None
        x_in_58 = torch.conv2d(
            x_98,
            l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_98 = l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_14 = x_in_58[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_14 = x_in_58[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_58 = None
        resid_14 = resid_13 + out1_14
        resid_13 = out1_14 = None
        dense_14 = torch.cat([dense_13, out2_14], dim=1)
        dense_13 = out2_14 = None
        x_in_59 = torch.cat((resid_14, dense_14), dim=1)
        x_99 = torch.nn.functional.batch_norm(
            x_in_59,
            l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_59 = l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        x_in_60 = torch.conv2d(
            x_100,
            l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_in_60,
            l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_60 = l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_102 = torch.nn.functional.relu(x_101, inplace=True)
        x_101 = None
        x_in_61 = torch.conv2d(
            x_102,
            l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_102 = l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_in_61,
            l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_61 = l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_in_62 = torch.conv2d(
            x_104,
            l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_15 = x_in_62[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_15 = x_in_62[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_62 = None
        resid_15 = resid_14 + out1_15
        resid_14 = out1_15 = None
        dense_15 = torch.cat([dense_14, out2_15], dim=1)
        dense_14 = out2_15 = None
        x_in_63 = torch.cat((resid_15, dense_15), dim=1)
        x_105 = torch.nn.functional.batch_norm(
            x_in_63,
            l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_63 = l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_in_64 = torch.conv2d(
            x_106,
            l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_107 = torch.nn.functional.batch_norm(
            x_in_64,
            l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_64 = l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_108 = torch.nn.functional.relu(x_107, inplace=True)
        x_107 = None
        x_in_65 = torch.conv2d(
            x_108,
            l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_108 = l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_in_65,
            l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_65 = l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_110 = torch.nn.functional.relu(x_109, inplace=True)
        x_109 = None
        x_in_66 = torch.conv2d(
            x_110,
            l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_110 = l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_16 = x_in_66[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_16 = x_in_66[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_66 = None
        resid_16 = resid_15 + out1_16
        resid_15 = out1_16 = None
        dense_16 = torch.cat([dense_15, out2_16], dim=1)
        dense_15 = out2_16 = None
        x_in_67 = torch.cat((resid_16, dense_16), dim=1)
        x_111 = torch.nn.functional.batch_norm(
            x_in_67,
            l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_67 = l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_in_68 = torch.conv2d(
            x_112,
            l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
            x_in_68,
            l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_68 = l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_114 = torch.nn.functional.relu(x_113, inplace=True)
        x_113 = None
        x_in_69 = torch.conv2d(
            x_114,
            l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_114 = l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_115 = torch.nn.functional.batch_norm(
            x_in_69,
            l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_69 = l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_116 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        x_in_70 = torch.conv2d(
            x_116,
            l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_17 = x_in_70[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_17 = x_in_70[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_70 = None
        resid_17 = resid_16 + out1_17
        resid_16 = out1_17 = None
        dense_17 = torch.cat([dense_16, out2_17], dim=1)
        dense_16 = out2_17 = None
        x_in_71 = torch.cat((resid_17, dense_17), dim=1)
        x_117 = torch.nn.functional.batch_norm(
            x_in_71,
            l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_71 = l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_118 = torch.nn.functional.relu(x_117, inplace=True)
        x_117 = None
        x_in_72 = torch.conv2d(
            x_118,
            l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_features_modules_conv4_7_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_in_72,
            l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_72 = l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_120 = torch.nn.functional.relu(x_119, inplace=True)
        x_119 = None
        x_in_73 = torch.conv2d(
            x_120,
            l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_120 = l_self_modules_features_modules_conv4_7_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_121 = torch.nn.functional.batch_norm(
            x_in_73,
            l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_73 = l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_122 = torch.nn.functional.relu(x_121, inplace=True)
        x_121 = None
        x_in_74 = torch.conv2d(
            x_122,
            l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_features_modules_conv4_7_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_18 = x_in_74[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_18 = x_in_74[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_74 = None
        resid_18 = resid_17 + out1_18
        resid_17 = out1_18 = None
        dense_18 = torch.cat([dense_17, out2_18], dim=1)
        dense_17 = out2_18 = None
        x_in_75 = torch.cat((resid_18, dense_18), dim=1)
        x_123 = torch.nn.functional.batch_norm(
            x_in_75,
            l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_75 = l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_124 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        x_in_76 = torch.conv2d(
            x_124,
            l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_features_modules_conv4_8_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_125 = torch.nn.functional.batch_norm(
            x_in_76,
            l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_76 = l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_126 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        x_in_77 = torch.conv2d(
            x_126,
            l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_126 = l_self_modules_features_modules_conv4_8_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_in_77,
            l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_77 = l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_128 = torch.nn.functional.relu(x_127, inplace=True)
        x_127 = None
        x_in_78 = torch.conv2d(
            x_128,
            l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_128 = l_self_modules_features_modules_conv4_8_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_19 = x_in_78[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_19 = x_in_78[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_78 = None
        resid_19 = resid_18 + out1_19
        resid_18 = out1_19 = None
        dense_19 = torch.cat([dense_18, out2_19], dim=1)
        dense_18 = out2_19 = None
        x_in_79 = torch.cat((resid_19, dense_19), dim=1)
        x_129 = torch.nn.functional.batch_norm(
            x_in_79,
            l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_79 = l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        x_in_80 = torch.conv2d(
            x_130,
            l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_features_modules_conv4_9_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_131 = torch.nn.functional.batch_norm(
            x_in_80,
            l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_80 = l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_132 = torch.nn.functional.relu(x_131, inplace=True)
        x_131 = None
        x_in_81 = torch.conv2d(
            x_132,
            l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_132 = l_self_modules_features_modules_conv4_9_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_133 = torch.nn.functional.batch_norm(
            x_in_81,
            l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_81 = l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_in_82 = torch.conv2d(
            x_134,
            l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_features_modules_conv4_9_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_20 = x_in_82[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_20 = x_in_82[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_82 = None
        resid_20 = resid_19 + out1_20
        resid_19 = out1_20 = None
        dense_20 = torch.cat([dense_19, out2_20], dim=1)
        dense_19 = out2_20 = None
        x_in_83 = torch.cat((resid_20, dense_20), dim=1)
        x_135 = torch.nn.functional.batch_norm(
            x_in_83,
            l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_83 = l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_136 = torch.nn.functional.relu(x_135, inplace=True)
        x_135 = None
        x_in_84 = torch.conv2d(
            x_136,
            l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_features_modules_conv4_10_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_in_84,
            l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_84 = l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_138 = torch.nn.functional.relu(x_137, inplace=True)
        x_137 = None
        x_in_85 = torch.conv2d(
            x_138,
            l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_138 = l_self_modules_features_modules_conv4_10_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_139 = torch.nn.functional.batch_norm(
            x_in_85,
            l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_85 = l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_140 = torch.nn.functional.relu(x_139, inplace=True)
        x_139 = None
        x_in_86 = torch.conv2d(
            x_140,
            l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_features_modules_conv4_10_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_21 = x_in_86[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_21 = x_in_86[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_86 = None
        resid_21 = resid_20 + out1_21
        resid_20 = out1_21 = None
        dense_21 = torch.cat([dense_20, out2_21], dim=1)
        dense_20 = out2_21 = None
        x_in_87 = torch.cat((resid_21, dense_21), dim=1)
        x_141 = torch.nn.functional.batch_norm(
            x_in_87,
            l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_87 = l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_142 = torch.nn.functional.relu(x_141, inplace=True)
        x_141 = None
        x_in_88 = torch.conv2d(
            x_142,
            l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_features_modules_conv4_11_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_in_88,
            l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_88 = l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        x_in_89 = torch.conv2d(
            x_144,
            l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_144 = l_self_modules_features_modules_conv4_11_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_145 = torch.nn.functional.batch_norm(
            x_in_89,
            l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_89 = l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_146 = torch.nn.functional.relu(x_145, inplace=True)
        x_145 = None
        x_in_90 = torch.conv2d(
            x_146,
            l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_features_modules_conv4_11_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_22 = x_in_90[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_22 = x_in_90[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_90 = None
        resid_22 = resid_21 + out1_22
        resid_21 = out1_22 = None
        dense_22 = torch.cat([dense_21, out2_22], dim=1)
        dense_21 = out2_22 = None
        x_in_91 = torch.cat((resid_22, dense_22), dim=1)
        x_147 = torch.nn.functional.batch_norm(
            x_in_91,
            l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_91 = l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_148 = torch.nn.functional.relu(x_147, inplace=True)
        x_147 = None
        x_in_92 = torch.conv2d(
            x_148,
            l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_features_modules_conv4_12_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_in_92,
            l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_92 = l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        x_in_93 = torch.conv2d(
            x_150,
            l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_150 = l_self_modules_features_modules_conv4_12_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_151 = torch.nn.functional.batch_norm(
            x_in_93,
            l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_93 = l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_152 = torch.nn.functional.relu(x_151, inplace=True)
        x_151 = None
        x_in_94 = torch.conv2d(
            x_152,
            l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_features_modules_conv4_12_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_23 = x_in_94[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_23 = x_in_94[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_94 = None
        resid_23 = resid_22 + out1_23
        resid_22 = out1_23 = None
        dense_23 = torch.cat([dense_22, out2_23], dim=1)
        dense_22 = out2_23 = None
        x_in_95 = torch.cat((resid_23, dense_23), dim=1)
        x_153 = torch.nn.functional.batch_norm(
            x_in_95,
            l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_95 = l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_154 = torch.nn.functional.relu(x_153, inplace=True)
        x_153 = None
        x_in_96 = torch.conv2d(
            x_154,
            l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_features_modules_conv4_13_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_in_96,
            l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_96 = l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        x_in_97 = torch.conv2d(
            x_156,
            l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_156 = l_self_modules_features_modules_conv4_13_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_157 = torch.nn.functional.batch_norm(
            x_in_97,
            l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_97 = l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_158 = torch.nn.functional.relu(x_157, inplace=True)
        x_157 = None
        x_in_98 = torch.conv2d(
            x_158,
            l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_158 = l_self_modules_features_modules_conv4_13_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_24 = x_in_98[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_24 = x_in_98[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_98 = None
        resid_24 = resid_23 + out1_24
        resid_23 = out1_24 = None
        dense_24 = torch.cat([dense_23, out2_24], dim=1)
        dense_23 = out2_24 = None
        x_in_99 = torch.cat((resid_24, dense_24), dim=1)
        x_159 = torch.nn.functional.batch_norm(
            x_in_99,
            l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_99 = l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_160 = torch.nn.functional.relu(x_159, inplace=True)
        x_159 = None
        x_in_100 = torch.conv2d(
            x_160,
            l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_features_modules_conv4_14_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_161 = torch.nn.functional.batch_norm(
            x_in_100,
            l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_100 = l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_162 = torch.nn.functional.relu(x_161, inplace=True)
        x_161 = None
        x_in_101 = torch.conv2d(
            x_162,
            l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_162 = l_self_modules_features_modules_conv4_14_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_163 = torch.nn.functional.batch_norm(
            x_in_101,
            l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_101 = l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_164 = torch.nn.functional.relu(x_163, inplace=True)
        x_163 = None
        x_in_102 = torch.conv2d(
            x_164,
            l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_164 = l_self_modules_features_modules_conv4_14_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_25 = x_in_102[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_25 = x_in_102[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_102 = None
        resid_25 = resid_24 + out1_25
        resid_24 = out1_25 = None
        dense_25 = torch.cat([dense_24, out2_25], dim=1)
        dense_24 = out2_25 = None
        x_in_103 = torch.cat((resid_25, dense_25), dim=1)
        x_165 = torch.nn.functional.batch_norm(
            x_in_103,
            l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_103 = l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        x_in_104 = torch.conv2d(
            x_166,
            l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_166 = l_self_modules_features_modules_conv4_15_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_167 = torch.nn.functional.batch_norm(
            x_in_104,
            l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_104 = l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_168 = torch.nn.functional.relu(x_167, inplace=True)
        x_167 = None
        x_in_105 = torch.conv2d(
            x_168,
            l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_168 = l_self_modules_features_modules_conv4_15_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_169 = torch.nn.functional.batch_norm(
            x_in_105,
            l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_105 = l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_170 = torch.nn.functional.relu(x_169, inplace=True)
        x_169 = None
        x_in_106 = torch.conv2d(
            x_170,
            l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_170 = l_self_modules_features_modules_conv4_15_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_26 = x_in_106[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_26 = x_in_106[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_106 = None
        resid_26 = resid_25 + out1_26
        resid_25 = out1_26 = None
        dense_26 = torch.cat([dense_25, out2_26], dim=1)
        dense_25 = out2_26 = None
        x_in_107 = torch.cat((resid_26, dense_26), dim=1)
        x_171 = torch.nn.functional.batch_norm(
            x_in_107,
            l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_107 = l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_172 = torch.nn.functional.relu(x_171, inplace=True)
        x_171 = None
        x_in_108 = torch.conv2d(
            x_172,
            l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_features_modules_conv4_16_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_173 = torch.nn.functional.batch_norm(
            x_in_108,
            l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_108 = l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_174 = torch.nn.functional.relu(x_173, inplace=True)
        x_173 = None
        x_in_109 = torch.conv2d(
            x_174,
            l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_174 = l_self_modules_features_modules_conv4_16_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_175 = torch.nn.functional.batch_norm(
            x_in_109,
            l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_109 = l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_176 = torch.nn.functional.relu(x_175, inplace=True)
        x_175 = None
        x_in_110 = torch.conv2d(
            x_176,
            l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_176 = l_self_modules_features_modules_conv4_16_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_27 = x_in_110[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_27 = x_in_110[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_110 = None
        resid_27 = resid_26 + out1_27
        resid_26 = out1_27 = None
        dense_27 = torch.cat([dense_26, out2_27], dim=1)
        dense_26 = out2_27 = None
        x_in_111 = torch.cat((resid_27, dense_27), dim=1)
        x_177 = torch.nn.functional.batch_norm(
            x_in_111,
            l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_111 = l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_178 = torch.nn.functional.relu(x_177, inplace=True)
        x_177 = None
        x_in_112 = torch.conv2d(
            x_178,
            l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_178 = l_self_modules_features_modules_conv4_17_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_179 = torch.nn.functional.batch_norm(
            x_in_112,
            l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_112 = l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_180 = torch.nn.functional.relu(x_179, inplace=True)
        x_179 = None
        x_in_113 = torch.conv2d(
            x_180,
            l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_180 = l_self_modules_features_modules_conv4_17_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_in_113,
            l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_113 = l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_182 = torch.nn.functional.relu(x_181, inplace=True)
        x_181 = None
        x_in_114 = torch.conv2d(
            x_182,
            l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_features_modules_conv4_17_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_28 = x_in_114[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_28 = x_in_114[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_114 = None
        resid_28 = resid_27 + out1_28
        resid_27 = out1_28 = None
        dense_28 = torch.cat([dense_27, out2_28], dim=1)
        dense_27 = out2_28 = None
        x_in_115 = torch.cat((resid_28, dense_28), dim=1)
        x_183 = torch.nn.functional.batch_norm(
            x_in_115,
            l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_115 = l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_184 = torch.nn.functional.relu(x_183, inplace=True)
        x_183 = None
        x_in_116 = torch.conv2d(
            x_184,
            l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_184 = l_self_modules_features_modules_conv4_18_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_185 = torch.nn.functional.batch_norm(
            x_in_116,
            l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_116 = l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        x_in_117 = torch.conv2d(
            x_186,
            l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_186 = l_self_modules_features_modules_conv4_18_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_187 = torch.nn.functional.batch_norm(
            x_in_117,
            l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_117 = l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_188 = torch.nn.functional.relu(x_187, inplace=True)
        x_187 = None
        x_in_118 = torch.conv2d(
            x_188,
            l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_features_modules_conv4_18_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_29 = x_in_118[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_29 = x_in_118[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_118 = None
        resid_29 = resid_28 + out1_29
        resid_28 = out1_29 = None
        dense_29 = torch.cat([dense_28, out2_29], dim=1)
        dense_28 = out2_29 = None
        x_in_119 = torch.cat((resid_29, dense_29), dim=1)
        x_189 = torch.nn.functional.batch_norm(
            x_in_119,
            l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_119 = l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_190 = torch.nn.functional.relu(x_189, inplace=True)
        x_189 = None
        x_in_120 = torch.conv2d(
            x_190,
            l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_features_modules_conv4_19_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_191 = torch.nn.functional.batch_norm(
            x_in_120,
            l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_120 = l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_192 = torch.nn.functional.relu(x_191, inplace=True)
        x_191 = None
        x_in_121 = torch.conv2d(
            x_192,
            l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_192 = l_self_modules_features_modules_conv4_19_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_193 = torch.nn.functional.batch_norm(
            x_in_121,
            l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_121 = l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_194 = torch.nn.functional.relu(x_193, inplace=True)
        x_193 = None
        x_in_122 = torch.conv2d(
            x_194,
            l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_194 = l_self_modules_features_modules_conv4_19_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_30 = x_in_122[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_30 = x_in_122[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_122 = None
        resid_30 = resid_29 + out1_30
        resid_29 = out1_30 = None
        dense_30 = torch.cat([dense_29, out2_30], dim=1)
        dense_29 = out2_30 = None
        x_in_123 = torch.cat((resid_30, dense_30), dim=1)
        x_195 = torch.nn.functional.batch_norm(
            x_in_123,
            l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_123 = l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_196 = torch.nn.functional.relu(x_195, inplace=True)
        x_195 = None
        x_in_124 = torch.conv2d(
            x_196,
            l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_196 = l_self_modules_features_modules_conv4_20_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_197 = torch.nn.functional.batch_norm(
            x_in_124,
            l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_124 = l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_198 = torch.nn.functional.relu(x_197, inplace=True)
        x_197 = None
        x_in_125 = torch.conv2d(
            x_198,
            l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_198 = l_self_modules_features_modules_conv4_20_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_199 = torch.nn.functional.batch_norm(
            x_in_125,
            l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_125 = l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_200 = torch.nn.functional.relu(x_199, inplace=True)
        x_199 = None
        x_in_126 = torch.conv2d(
            x_200,
            l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_200 = l_self_modules_features_modules_conv4_20_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_31 = x_in_126[
            (
                slice(None, None, None),
                slice(None, 1024, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_31 = x_in_126[
            (
                slice(None, None, None),
                slice(1024, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_126 = None
        resid_31 = resid_30 + out1_31
        resid_30 = out1_31 = None
        dense_31 = torch.cat([dense_30, out2_31], dim=1)
        dense_30 = out2_31 = None
        x_in_127 = torch.cat((resid_31, dense_31), dim=1)
        resid_31 = dense_31 = None
        x_201 = torch.nn.functional.batch_norm(
            x_in_127,
            l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_bias_ = (None)
        x_202 = torch.nn.functional.relu(x_201, inplace=True)
        x_201 = None
        x_s_3 = torch.conv2d(
            x_202,
            l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_202 = l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_conv_parameters_weight_ = (None)
        x_s1_3 = x_s_3[
            (
                slice(None, None, None),
                slice(None, 2048, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s2_3 = x_s_3[
            (
                slice(None, None, None),
                slice(2048, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s_3 = None
        x_203 = torch.nn.functional.batch_norm(
            x_in_127,
            l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_127 = l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_204 = torch.nn.functional.relu(x_203, inplace=True)
        x_203 = None
        x_in_128 = torch.conv2d(
            x_204,
            l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_204 = l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_205 = torch.nn.functional.batch_norm(
            x_in_128,
            l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_128 = l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_206 = torch.nn.functional.relu(x_205, inplace=True)
        x_205 = None
        x_in_129 = torch.conv2d(
            x_206,
            l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            50,
        )
        x_206 = l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_207 = torch.nn.functional.batch_norm(
            x_in_129,
            l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_129 = l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_208 = torch.nn.functional.relu(x_207, inplace=True)
        x_207 = None
        x_in_130 = torch.conv2d(
            x_208,
            l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_32 = x_in_130[
            (
                slice(None, None, None),
                slice(None, 2048, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_32 = x_in_130[
            (
                slice(None, None, None),
                slice(2048, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_130 = None
        resid_32 = x_s1_3 + out1_32
        x_s1_3 = out1_32 = None
        dense_32 = torch.cat([x_s2_3, out2_32], dim=1)
        x_s2_3 = out2_32 = None
        x_in_131 = torch.cat((resid_32, dense_32), dim=1)
        x_209 = torch.nn.functional.batch_norm(
            x_in_131,
            l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_131 = l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_210 = torch.nn.functional.relu(x_209, inplace=True)
        x_209 = None
        x_in_132 = torch.conv2d(
            x_210,
            l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_210 = l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_211 = torch.nn.functional.batch_norm(
            x_in_132,
            l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_132 = l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_212 = torch.nn.functional.relu(x_211, inplace=True)
        x_211 = None
        x_in_133 = torch.conv2d(
            x_212,
            l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_212 = l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_213 = torch.nn.functional.batch_norm(
            x_in_133,
            l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_133 = l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_214 = torch.nn.functional.relu(x_213, inplace=True)
        x_213 = None
        x_in_134 = torch.conv2d(
            x_214,
            l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_214 = l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_33 = x_in_134[
            (
                slice(None, None, None),
                slice(None, 2048, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_33 = x_in_134[
            (
                slice(None, None, None),
                slice(2048, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_134 = None
        resid_33 = resid_32 + out1_33
        resid_32 = out1_33 = None
        dense_33 = torch.cat([dense_32, out2_33], dim=1)
        dense_32 = out2_33 = None
        x_in_135 = torch.cat((resid_33, dense_33), dim=1)
        x_215 = torch.nn.functional.batch_norm(
            x_in_135,
            l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_135 = l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_216 = torch.nn.functional.relu(x_215, inplace=True)
        x_215 = None
        x_in_136 = torch.conv2d(
            x_216,
            l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_216 = l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_217 = torch.nn.functional.batch_norm(
            x_in_136,
            l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_136 = l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_218 = torch.nn.functional.relu(x_217, inplace=True)
        x_217 = None
        x_in_137 = torch.conv2d(
            x_218,
            l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            50,
        )
        x_218 = l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_219 = torch.nn.functional.batch_norm(
            x_in_137,
            l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_137 = l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_220 = torch.nn.functional.relu(x_219, inplace=True)
        x_219 = None
        x_in_138 = torch.conv2d(
            x_220,
            l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_220 = l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_conv_parameters_weight_ = (None)
        out1_34 = x_in_138[
            (
                slice(None, None, None),
                slice(None, 2048, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out2_34 = x_in_138[
            (
                slice(None, None, None),
                slice(2048, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_in_138 = None
        resid_34 = resid_33 + out1_34
        resid_33 = out1_34 = None
        dense_34 = torch.cat([dense_33, out2_34], dim=1)
        dense_33 = out2_34 = None
        x_221 = torch.cat((resid_34, dense_34), dim=1)
        resid_34 = dense_34 = None
        x_222 = torch.nn.functional.batch_norm(
            x_221,
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_221 = (
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_bias_
        ) = None
        x_223 = torch.nn.functional.relu(x_222, inplace=False)
        x_222 = None
        x_224 = torch.nn.functional.adaptive_avg_pool2d(x_223, 1)
        x_223 = None
        x_225 = torch.conv2d(
            x_224,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_224 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        x_226 = x_225.flatten(1, -1)
        x_225 = None
        return (x_226,)
