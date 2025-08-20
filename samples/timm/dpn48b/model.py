import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_conv1_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv2_1_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_1_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv2_2_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_2_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv2_3_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_3_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv3_1_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_1_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv3_2_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_2_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv3_3_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_3_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv3_4_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv3_4_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv4_1_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_1_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv4_2_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_2_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv4_3_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_3_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv4_4_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_4_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv4_5_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_5_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv4_6_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv4_6_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv5_1_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_1_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv5_2_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_2_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_conv5_3_modules_c1x1_c1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv5_3_modules_c1x1_c2_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_conv2_1_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv2_1_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv2_1_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv2_1_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv2_2_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv2_2_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv2_2_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv2_2_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv2_3_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv2_3_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv2_3_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv2_3_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv3_1_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv3_1_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv3_1_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv3_1_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv3_2_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv3_2_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv3_2_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv3_2_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv3_3_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv3_3_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv3_3_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv3_3_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv3_4_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv3_4_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv3_4_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv3_4_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv4_1_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv4_1_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv4_1_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv4_1_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv4_2_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv4_2_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv4_2_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv4_2_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv4_3_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv4_3_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv4_3_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv4_3_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv4_4_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv4_4_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv4_4_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv4_4_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv4_5_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv4_5_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv4_5_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv4_5_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv4_6_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv4_6_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv4_6_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv4_6_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv5_1_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv5_1_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv5_1_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv5_1_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv5_2_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv5_2_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv5_2_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv5_2_modules_c1x1_c2_parameters_weight_
        )
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
        l_self_modules_features_modules_conv5_3_modules_c1x1_c1_parameters_weight_ = (
            L_self_modules_features_modules_conv5_3_modules_c1x1_c1_parameters_weight_
        )
        l_self_modules_features_modules_conv5_3_modules_c1x1_c2_parameters_weight_ = (
            L_self_modules_features_modules_conv5_3_modules_c1x1_c2_parameters_weight_
        )
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
            (1, 1),
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
        x_2 = torch.nn.functional.silu(x_1, inplace=True)
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
                slice(None, 64, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s2 = x_s[
            (
                slice(None, None, None),
                slice(64, None, None),
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
            32,
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
        out1 = torch.conv2d(
            x_10,
            l_self_modules_features_modules_conv2_1_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv2_1_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2 = torch.conv2d(
            x_10,
            l_self_modules_features_modules_conv2_1_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = (
            l_self_modules_features_modules_conv2_1_modules_c1x1_c2_parameters_weight_
        ) = None
        resid = x_s1 + out1
        x_s1 = out1 = None
        dense = torch.cat([x_s2, out2], dim=1)
        x_s2 = out2 = None
        x_in_2 = torch.cat((resid, dense), dim=1)
        x_11 = torch.nn.functional.batch_norm(
            x_in_2,
            l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_2 = l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_2_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        x_in_3 = torch.conv2d(
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
            x_in_3,
            l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_3 = l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        x_in_4 = torch.conv2d(
            x_14,
            l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_14 = l_self_modules_features_modules_conv2_2_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_in_4,
            l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_4 = l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_2_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_16 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        out1_1 = torch.conv2d(
            x_16,
            l_self_modules_features_modules_conv2_2_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv2_2_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_1 = torch.conv2d(
            x_16,
            l_self_modules_features_modules_conv2_2_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = (
            l_self_modules_features_modules_conv2_2_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_1 = resid + out1_1
        resid = out1_1 = None
        dense_1 = torch.cat([dense, out2_1], dim=1)
        dense = out2_1 = None
        x_in_5 = torch.cat((resid_1, dense_1), dim=1)
        x_17 = torch.nn.functional.batch_norm(
            x_in_5,
            l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_5 = l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_3_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        x_in_6 = torch.conv2d(
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
            x_in_6,
            l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_6 = l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_in_7 = torch.conv2d(
            x_20,
            l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_20 = l_self_modules_features_modules_conv2_3_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_21 = torch.nn.functional.batch_norm(
            x_in_7,
            l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_7 = l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv2_3_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        out1_2 = torch.conv2d(
            x_22,
            l_self_modules_features_modules_conv2_3_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv2_3_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_2 = torch.conv2d(
            x_22,
            l_self_modules_features_modules_conv2_3_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = (
            l_self_modules_features_modules_conv2_3_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_2 = resid_1 + out1_2
        resid_1 = out1_2 = None
        dense_2 = torch.cat([dense_1, out2_2], dim=1)
        dense_1 = out2_2 = None
        x_in_8 = torch.cat((resid_2, dense_2), dim=1)
        resid_2 = dense_2 = None
        x_23 = torch.nn.functional.batch_norm(
            x_in_8,
            l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_bn_parameters_bias_ = (None)
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        x_s_1 = torch.conv2d(
            x_24,
            l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_24 = l_self_modules_features_modules_conv3_1_modules_c1x1_w_s2_modules_conv_parameters_weight_ = (None)
        x_s1_1 = x_s_1[
            (
                slice(None, None, None),
                slice(None, 128, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s2_1 = x_s_1[
            (
                slice(None, None, None),
                slice(128, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s_1 = None
        x_25 = torch.nn.functional.batch_norm(
            x_in_8,
            l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_8 = l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_26 = torch.nn.functional.relu(x_25, inplace=True)
        x_25 = None
        x_in_9 = torch.conv2d(
            x_26,
            l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_features_modules_conv3_1_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_in_9,
            l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_9 = l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.relu(x_27, inplace=True)
        x_27 = None
        x_in_10 = torch.conv2d(
            x_28,
            l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        x_28 = l_self_modules_features_modules_conv3_1_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_in_10,
            l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_10 = l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_1_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        out1_3 = torch.conv2d(
            x_30,
            l_self_modules_features_modules_conv3_1_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv3_1_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_3 = torch.conv2d(
            x_30,
            l_self_modules_features_modules_conv3_1_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = (
            l_self_modules_features_modules_conv3_1_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_3 = x_s1_1 + out1_3
        x_s1_1 = out1_3 = None
        dense_3 = torch.cat([x_s2_1, out2_3], dim=1)
        x_s2_1 = out2_3 = None
        x_in_11 = torch.cat((resid_3, dense_3), dim=1)
        x_31 = torch.nn.functional.batch_norm(
            x_in_11,
            l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_11 = l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x_in_12 = torch.conv2d(
            x_32,
            l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_features_modules_conv3_2_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_in_12,
            l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_12 = l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        x_in_13 = torch.conv2d(
            x_34,
            l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_34 = l_self_modules_features_modules_conv3_2_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_in_13,
            l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_13 = l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_2_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_36 = torch.nn.functional.relu(x_35, inplace=True)
        x_35 = None
        out1_4 = torch.conv2d(
            x_36,
            l_self_modules_features_modules_conv3_2_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv3_2_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_4 = torch.conv2d(
            x_36,
            l_self_modules_features_modules_conv3_2_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_36 = (
            l_self_modules_features_modules_conv3_2_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_4 = resid_3 + out1_4
        resid_3 = out1_4 = None
        dense_4 = torch.cat([dense_3, out2_4], dim=1)
        dense_3 = out2_4 = None
        x_in_14 = torch.cat((resid_4, dense_4), dim=1)
        x_37 = torch.nn.functional.batch_norm(
            x_in_14,
            l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_14 = l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_in_15 = torch.conv2d(
            x_38,
            l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_features_modules_conv3_3_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_in_15,
            l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_15 = l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_in_16 = torch.conv2d(
            x_40,
            l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_40 = l_self_modules_features_modules_conv3_3_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_in_16,
            l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_16 = l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_3_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        out1_5 = torch.conv2d(
            x_42,
            l_self_modules_features_modules_conv3_3_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv3_3_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_5 = torch.conv2d(
            x_42,
            l_self_modules_features_modules_conv3_3_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = (
            l_self_modules_features_modules_conv3_3_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_5 = resid_4 + out1_5
        resid_4 = out1_5 = None
        dense_5 = torch.cat([dense_4, out2_5], dim=1)
        dense_4 = out2_5 = None
        x_in_17 = torch.cat((resid_5, dense_5), dim=1)
        x_43 = torch.nn.functional.batch_norm(
            x_in_17,
            l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_17 = l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_44 = torch.nn.functional.relu(x_43, inplace=True)
        x_43 = None
        x_in_18 = torch.conv2d(
            x_44,
            l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_features_modules_conv3_4_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_in_18,
            l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_18 = l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_in_19 = torch.conv2d(
            x_46,
            l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_46 = l_self_modules_features_modules_conv3_4_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_in_19,
            l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_19 = l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv3_4_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_48 = torch.nn.functional.relu(x_47, inplace=True)
        x_47 = None
        out1_6 = torch.conv2d(
            x_48,
            l_self_modules_features_modules_conv3_4_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv3_4_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_6 = torch.conv2d(
            x_48,
            l_self_modules_features_modules_conv3_4_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = (
            l_self_modules_features_modules_conv3_4_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_6 = resid_5 + out1_6
        resid_5 = out1_6 = None
        dense_6 = torch.cat([dense_5, out2_6], dim=1)
        dense_5 = out2_6 = None
        x_in_20 = torch.cat((resid_6, dense_6), dim=1)
        resid_6 = dense_6 = None
        x_49 = torch.nn.functional.batch_norm(
            x_in_20,
            l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_bn_parameters_bias_ = (None)
        x_50 = torch.nn.functional.relu(x_49, inplace=True)
        x_49 = None
        x_s_2 = torch.conv2d(
            x_50,
            l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_50 = l_self_modules_features_modules_conv4_1_modules_c1x1_w_s2_modules_conv_parameters_weight_ = (None)
        x_s1_2 = x_s_2[
            (
                slice(None, None, None),
                slice(None, 256, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s2_2 = x_s_2[
            (
                slice(None, None, None),
                slice(256, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s_2 = None
        x_51 = torch.nn.functional.batch_norm(
            x_in_20,
            l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_20 = l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_in_21 = torch.conv2d(
            x_52,
            l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_features_modules_conv4_1_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_in_21,
            l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_21 = l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_54 = torch.nn.functional.relu(x_53, inplace=True)
        x_53 = None
        x_in_22 = torch.conv2d(
            x_54,
            l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        x_54 = l_self_modules_features_modules_conv4_1_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
            x_in_22,
            l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_22 = l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_1_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        out1_7 = torch.conv2d(
            x_56,
            l_self_modules_features_modules_conv4_1_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv4_1_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_7 = torch.conv2d(
            x_56,
            l_self_modules_features_modules_conv4_1_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_56 = (
            l_self_modules_features_modules_conv4_1_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_7 = x_s1_2 + out1_7
        x_s1_2 = out1_7 = None
        dense_7 = torch.cat([x_s2_2, out2_7], dim=1)
        x_s2_2 = out2_7 = None
        x_in_23 = torch.cat((resid_7, dense_7), dim=1)
        x_57 = torch.nn.functional.batch_norm(
            x_in_23,
            l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_23 = l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        x_in_24 = torch.conv2d(
            x_58,
            l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_features_modules_conv4_2_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_in_24,
            l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_24 = l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_in_25 = torch.conv2d(
            x_60,
            l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_60 = l_self_modules_features_modules_conv4_2_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_61 = torch.nn.functional.batch_norm(
            x_in_25,
            l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_25 = l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_2_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_62 = torch.nn.functional.relu(x_61, inplace=True)
        x_61 = None
        out1_8 = torch.conv2d(
            x_62,
            l_self_modules_features_modules_conv4_2_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv4_2_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_8 = torch.conv2d(
            x_62,
            l_self_modules_features_modules_conv4_2_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = (
            l_self_modules_features_modules_conv4_2_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_8 = resid_7 + out1_8
        resid_7 = out1_8 = None
        dense_8 = torch.cat([dense_7, out2_8], dim=1)
        dense_7 = out2_8 = None
        x_in_26 = torch.cat((resid_8, dense_8), dim=1)
        x_63 = torch.nn.functional.batch_norm(
            x_in_26,
            l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_26 = l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        x_in_27 = torch.conv2d(
            x_64,
            l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_features_modules_conv4_3_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_in_27,
            l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_27 = l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_66 = torch.nn.functional.relu(x_65, inplace=True)
        x_65 = None
        x_in_28 = torch.conv2d(
            x_66,
            l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_66 = l_self_modules_features_modules_conv4_3_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_in_28,
            l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_28 = l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_3_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        out1_9 = torch.conv2d(
            x_68,
            l_self_modules_features_modules_conv4_3_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv4_3_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_9 = torch.conv2d(
            x_68,
            l_self_modules_features_modules_conv4_3_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = (
            l_self_modules_features_modules_conv4_3_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_9 = resid_8 + out1_9
        resid_8 = out1_9 = None
        dense_9 = torch.cat([dense_8, out2_9], dim=1)
        dense_8 = out2_9 = None
        x_in_29 = torch.cat((resid_9, dense_9), dim=1)
        x_69 = torch.nn.functional.batch_norm(
            x_in_29,
            l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_29 = l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_70 = torch.nn.functional.relu(x_69, inplace=True)
        x_69 = None
        x_in_30 = torch.conv2d(
            x_70,
            l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_features_modules_conv4_4_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_in_30,
            l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_30 = l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_in_31 = torch.conv2d(
            x_72,
            l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_72 = l_self_modules_features_modules_conv4_4_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_in_31,
            l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_31 = l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_4_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        out1_10 = torch.conv2d(
            x_74,
            l_self_modules_features_modules_conv4_4_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv4_4_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_10 = torch.conv2d(
            x_74,
            l_self_modules_features_modules_conv4_4_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = (
            l_self_modules_features_modules_conv4_4_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_10 = resid_9 + out1_10
        resid_9 = out1_10 = None
        dense_10 = torch.cat([dense_9, out2_10], dim=1)
        dense_9 = out2_10 = None
        x_in_32 = torch.cat((resid_10, dense_10), dim=1)
        x_75 = torch.nn.functional.batch_norm(
            x_in_32,
            l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_32 = l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_76 = torch.nn.functional.relu(x_75, inplace=True)
        x_75 = None
        x_in_33 = torch.conv2d(
            x_76,
            l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_features_modules_conv4_5_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_in_33,
            l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_33 = l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        x_in_34 = torch.conv2d(
            x_78,
            l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_78 = l_self_modules_features_modules_conv4_5_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_in_34,
            l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_34 = l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_5_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_80 = torch.nn.functional.relu(x_79, inplace=True)
        x_79 = None
        out1_11 = torch.conv2d(
            x_80,
            l_self_modules_features_modules_conv4_5_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv4_5_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_11 = torch.conv2d(
            x_80,
            l_self_modules_features_modules_conv4_5_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_80 = (
            l_self_modules_features_modules_conv4_5_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_11 = resid_10 + out1_11
        resid_10 = out1_11 = None
        dense_11 = torch.cat([dense_10, out2_11], dim=1)
        dense_10 = out2_11 = None
        x_in_35 = torch.cat((resid_11, dense_11), dim=1)
        x_81 = torch.nn.functional.batch_norm(
            x_in_35,
            l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_35 = l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_in_36 = torch.conv2d(
            x_82,
            l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_features_modules_conv4_6_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_in_36,
            l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_36 = l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        x_in_37 = torch.conv2d(
            x_84,
            l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_84 = l_self_modules_features_modules_conv4_6_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_85 = torch.nn.functional.batch_norm(
            x_in_37,
            l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_37 = l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv4_6_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        out1_12 = torch.conv2d(
            x_86,
            l_self_modules_features_modules_conv4_6_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv4_6_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_12 = torch.conv2d(
            x_86,
            l_self_modules_features_modules_conv4_6_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = (
            l_self_modules_features_modules_conv4_6_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_12 = resid_11 + out1_12
        resid_11 = out1_12 = None
        dense_12 = torch.cat([dense_11, out2_12], dim=1)
        dense_11 = out2_12 = None
        x_in_38 = torch.cat((resid_12, dense_12), dim=1)
        resid_12 = dense_12 = None
        x_87 = torch.nn.functional.batch_norm(
            x_in_38,
            l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_bn_parameters_bias_ = (None)
        x_88 = torch.nn.functional.relu(x_87, inplace=True)
        x_87 = None
        x_s_3 = torch.conv2d(
            x_88,
            l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_features_modules_conv5_1_modules_c1x1_w_s2_modules_conv_parameters_weight_ = (None)
        x_s1_3 = x_s_3[
            (
                slice(None, None, None),
                slice(None, 512, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s2_3 = x_s_3[
            (
                slice(None, None, None),
                slice(512, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        x_s_3 = None
        x_89 = torch.nn.functional.batch_norm(
            x_in_38,
            l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_38 = l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_in_39 = torch.conv2d(
            x_90,
            l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_features_modules_conv5_1_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_91 = torch.nn.functional.batch_norm(
            x_in_39,
            l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_39 = l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_92 = torch.nn.functional.relu(x_91, inplace=True)
        x_91 = None
        x_in_40 = torch.conv2d(
            x_92,
            l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        x_92 = l_self_modules_features_modules_conv5_1_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_in_40,
            l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_40 = l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_1_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        out1_13 = torch.conv2d(
            x_94,
            l_self_modules_features_modules_conv5_1_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv5_1_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_13 = torch.conv2d(
            x_94,
            l_self_modules_features_modules_conv5_1_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = (
            l_self_modules_features_modules_conv5_1_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_13 = x_s1_3 + out1_13
        x_s1_3 = out1_13 = None
        dense_13 = torch.cat([x_s2_3, out2_13], dim=1)
        x_s2_3 = out2_13 = None
        x_in_41 = torch.cat((resid_13, dense_13), dim=1)
        x_95 = torch.nn.functional.batch_norm(
            x_in_41,
            l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_41 = l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        x_in_42 = torch.conv2d(
            x_96,
            l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_features_modules_conv5_2_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_97 = torch.nn.functional.batch_norm(
            x_in_42,
            l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_42 = l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_98 = torch.nn.functional.relu(x_97, inplace=True)
        x_97 = None
        x_in_43 = torch.conv2d(
            x_98,
            l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_98 = l_self_modules_features_modules_conv5_2_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_in_43,
            l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_43 = l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_2_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        out1_14 = torch.conv2d(
            x_100,
            l_self_modules_features_modules_conv5_2_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv5_2_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_14 = torch.conv2d(
            x_100,
            l_self_modules_features_modules_conv5_2_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = (
            l_self_modules_features_modules_conv5_2_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_14 = resid_13 + out1_14
        resid_13 = out1_14 = None
        dense_14 = torch.cat([dense_13, out2_14], dim=1)
        dense_13 = out2_14 = None
        x_in_44 = torch.cat((resid_14, dense_14), dim=1)
        x_101 = torch.nn.functional.batch_norm(
            x_in_44,
            l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_44 = l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_bn_parameters_bias_ = (None)
        x_102 = torch.nn.functional.relu(x_101, inplace=True)
        x_101 = None
        x_in_45 = torch.conv2d(
            x_102,
            l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_102 = l_self_modules_features_modules_conv5_3_modules_c1x1_a_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_in_45,
            l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_45 = l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_bn_parameters_bias_ = (None)
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_in_46 = torch.conv2d(
            x_104,
            l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_104 = l_self_modules_features_modules_conv5_3_modules_c3x3_b_modules_conv_parameters_weight_ = (None)
        x_105 = torch.nn.functional.batch_norm(
            x_in_46,
            l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_in_46 = l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_buffers_running_var_ = l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_weight_ = l_self_modules_features_modules_conv5_3_modules_c1x1_c_modules_bn_parameters_bias_ = (None)
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        out1_15 = torch.conv2d(
            x_106,
            l_self_modules_features_modules_conv5_3_modules_c1x1_c1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_conv5_3_modules_c1x1_c1_parameters_weight_ = (
            None
        )
        out2_15 = torch.conv2d(
            x_106,
            l_self_modules_features_modules_conv5_3_modules_c1x1_c2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = (
            l_self_modules_features_modules_conv5_3_modules_c1x1_c2_parameters_weight_
        ) = None
        resid_15 = resid_14 + out1_15
        resid_14 = out1_15 = None
        dense_15 = torch.cat([dense_14, out2_15], dim=1)
        dense_14 = out2_15 = None
        x_107 = torch.cat((resid_15, dense_15), dim=1)
        resid_15 = dense_15 = None
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_weight_,
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_107 = (
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_conv5_bn_ac_modules_bn_parameters_bias_
        ) = None
        x_109 = torch.nn.functional.silu(x_108, inplace=False)
        x_108 = None
        x_110 = torch.nn.functional.adaptive_avg_pool2d(x_109, 1)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_110 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        x_112 = x_111.flatten(1, -1)
        x_111 = None
        return (x_112,)
