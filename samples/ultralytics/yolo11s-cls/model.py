import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_model_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_7_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_7_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_7_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_7_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_10_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_linear_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_linear_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_model_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_0_modules_bn_momentum = (
            L_self_modules_model_modules_0_modules_bn_momentum
        )
        l_self_modules_model_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_0_modules_bn_eps = (
            L_self_modules_model_modules_0_modules_bn_eps
        )
        l_self_modules_model_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_1_modules_bn_momentum = (
            L_self_modules_model_modules_1_modules_bn_momentum
        )
        l_self_modules_model_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_1_modules_bn_eps = (
            L_self_modules_model_modules_1_modules_bn_eps
        )
        l_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_momentum = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_momentum
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_eps = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_eps
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_momentum = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_momentum
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_eps = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_eps
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_momentum = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_momentum
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_eps = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_eps
        l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_momentum = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_momentum
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_eps = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_eps
        )
        l_self_modules_model_modules_3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_3_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_3_modules_bn_momentum = (
            L_self_modules_model_modules_3_modules_bn_momentum
        )
        l_self_modules_model_modules_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_3_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_3_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_3_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_3_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_3_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_3_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_3_modules_bn_eps = (
            L_self_modules_model_modules_3_modules_bn_eps
        )
        l_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_momentum = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_momentum
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_eps = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_eps
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_momentum = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_momentum
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_eps = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_eps
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_momentum = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_momentum
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_eps = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_eps
        l_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_momentum = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_momentum
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_eps = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_eps
        )
        l_self_modules_model_modules_5_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_5_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_5_modules_bn_momentum = (
            L_self_modules_model_modules_5_modules_bn_momentum
        )
        l_self_modules_model_modules_5_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_5_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_5_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_5_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_5_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_5_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_5_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_5_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_5_modules_bn_eps = (
            L_self_modules_model_modules_5_modules_bn_eps
        )
        l_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_momentum = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_momentum
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_eps = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_eps
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_eps
        l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_momentum = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_momentum
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_eps = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_eps
        )
        l_self_modules_model_modules_7_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_7_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_7_modules_bn_momentum = (
            L_self_modules_model_modules_7_modules_bn_momentum
        )
        l_self_modules_model_modules_7_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_7_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_7_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_7_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_7_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_7_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_7_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_7_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_7_modules_bn_eps = (
            L_self_modules_model_modules_7_modules_bn_eps
        )
        l_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_momentum = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_momentum
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_eps = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_eps
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_eps
        l_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_momentum = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_momentum
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_eps = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_eps
        )
        l_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_cv1_modules_bn_momentum = (
            L_self_modules_model_modules_9_modules_cv1_modules_bn_momentum
        )
        l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_9_modules_cv1_modules_bn_eps = (
            L_self_modules_model_modules_9_modules_cv1_modules_bn_eps
        )
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_cv2_modules_bn_momentum = (
            L_self_modules_model_modules_9_modules_cv2_modules_bn_momentum
        )
        l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_9_modules_cv2_modules_bn_eps = (
            L_self_modules_model_modules_9_modules_cv2_modules_bn_eps
        )
        l_self_modules_model_modules_10_modules_conv_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_10_modules_conv_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_10_modules_conv_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_10_modules_conv_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_10_modules_conv_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_10_modules_conv_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_10_modules_linear_parameters_weight_ = (
            L_self_modules_model_modules_10_modules_linear_parameters_weight_
        )
        l_self_modules_model_modules_10_modules_linear_parameters_bias_ = (
            L_self_modules_model_modules_10_modules_linear_parameters_bias_
        )
        conv2d = torch.conv2d(
            l_x_,
            l_self_modules_model_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_model_modules_0_modules_conv_parameters_weight_ = None
        item = l_self_modules_model_modules_0_modules_bn_momentum.item()
        l_self_modules_model_modules_0_modules_bn_momentum = None
        item_1 = l_self_modules_model_modules_0_modules_bn_eps.item()
        l_self_modules_model_modules_0_modules_bn_eps = None
        batch_norm = torch.nn.functional.batch_norm(
            conv2d,
            l_self_modules_model_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_0_modules_bn_parameters_bias_,
            False,
            item,
            item_1,
        )
        conv2d = (
            l_self_modules_model_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_0_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_0_modules_bn_parameters_bias_
        ) = item = item_1 = None
        x = torch.nn.functional.silu(batch_norm, inplace=True)
        batch_norm = None
        conv2d_1 = torch.conv2d(
            x,
            l_self_modules_model_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x = l_self_modules_model_modules_1_modules_conv_parameters_weight_ = None
        item_2 = l_self_modules_model_modules_1_modules_bn_momentum.item()
        l_self_modules_model_modules_1_modules_bn_momentum = None
        item_3 = l_self_modules_model_modules_1_modules_bn_eps.item()
        l_self_modules_model_modules_1_modules_bn_eps = None
        batch_norm_1 = torch.nn.functional.batch_norm(
            conv2d_1,
            l_self_modules_model_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_1_modules_bn_parameters_bias_,
            False,
            item_2,
            item_3,
        )
        conv2d_1 = (
            l_self_modules_model_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_1_modules_bn_parameters_bias_
        ) = item_2 = item_3 = None
        x_1 = torch.nn.functional.silu(batch_norm_1, inplace=True)
        batch_norm_1 = None
        conv2d_2 = torch.conv2d(
            x_1,
            l_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_1 = (
            l_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_
        ) = None
        item_4 = l_self_modules_model_modules_2_modules_cv1_modules_bn_momentum.item()
        l_self_modules_model_modules_2_modules_cv1_modules_bn_momentum = None
        item_5 = l_self_modules_model_modules_2_modules_cv1_modules_bn_eps.item()
        l_self_modules_model_modules_2_modules_cv1_modules_bn_eps = None
        batch_norm_2 = torch.nn.functional.batch_norm(
            conv2d_2,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_4,
            item_5,
        )
        conv2d_2 = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_
        ) = item_4 = item_5 = None
        silu_2 = torch.nn.functional.silu(batch_norm_2, inplace=True)
        batch_norm_2 = None
        chunk = silu_2.chunk(2, 1)
        silu_2 = None
        getitem = chunk[0]
        getitem_1 = chunk[1]
        chunk = None
        conv2d_3 = torch.conv2d(
            getitem_1,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        item_6 = (
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_momentum = (
            None
        )
        item_7 = (
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_eps = (
            None
        )
        batch_norm_3 = torch.nn.functional.batch_norm(
            conv2d_3,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_6,
            item_7,
        )
        conv2d_3 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (item_6) = (
            item_7
        ) = None
        silu_3 = torch.nn.functional.silu(batch_norm_3, inplace=True)
        batch_norm_3 = None
        conv2d_4 = torch.conv2d(
            silu_3,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_3 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        item_8 = (
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_momentum = (
            None
        )
        item_9 = (
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_eps.item()
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_eps = (
            None
        )
        batch_norm_4 = torch.nn.functional.batch_norm(
            conv2d_4,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_8,
            item_9,
        )
        conv2d_4 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (item_8) = (
            item_9
        ) = None
        silu_4 = torch.nn.functional.silu(batch_norm_4, inplace=True)
        batch_norm_4 = None
        add = getitem_1 + silu_4
        silu_4 = None
        cat = torch.cat([getitem, getitem_1, add], 1)
        getitem = getitem_1 = add = None
        conv2d_5 = torch.conv2d(
            cat,
            l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = (
            l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_
        ) = None
        item_10 = l_self_modules_model_modules_2_modules_cv2_modules_bn_momentum.item()
        l_self_modules_model_modules_2_modules_cv2_modules_bn_momentum = None
        item_11 = l_self_modules_model_modules_2_modules_cv2_modules_bn_eps.item()
        l_self_modules_model_modules_2_modules_cv2_modules_bn_eps = None
        batch_norm_5 = torch.nn.functional.batch_norm(
            conv2d_5,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_10,
            item_11,
        )
        conv2d_5 = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_
        ) = item_10 = item_11 = None
        x_2 = torch.nn.functional.silu(batch_norm_5, inplace=True)
        batch_norm_5 = None
        conv2d_6 = torch.conv2d(
            x_2,
            l_self_modules_model_modules_3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_model_modules_3_modules_conv_parameters_weight_ = None
        item_12 = l_self_modules_model_modules_3_modules_bn_momentum.item()
        l_self_modules_model_modules_3_modules_bn_momentum = None
        item_13 = l_self_modules_model_modules_3_modules_bn_eps.item()
        l_self_modules_model_modules_3_modules_bn_eps = None
        batch_norm_6 = torch.nn.functional.batch_norm(
            conv2d_6,
            l_self_modules_model_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_3_modules_bn_parameters_bias_,
            False,
            item_12,
            item_13,
        )
        conv2d_6 = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_3_modules_bn_parameters_bias_
        ) = item_12 = item_13 = None
        x_3 = torch.nn.functional.silu(batch_norm_6, inplace=True)
        batch_norm_6 = None
        conv2d_7 = torch.conv2d(
            x_3,
            l_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = (
            l_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_
        ) = None
        item_14 = l_self_modules_model_modules_4_modules_cv1_modules_bn_momentum.item()
        l_self_modules_model_modules_4_modules_cv1_modules_bn_momentum = None
        item_15 = l_self_modules_model_modules_4_modules_cv1_modules_bn_eps.item()
        l_self_modules_model_modules_4_modules_cv1_modules_bn_eps = None
        batch_norm_7 = torch.nn.functional.batch_norm(
            conv2d_7,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_14,
            item_15,
        )
        conv2d_7 = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_
        ) = item_14 = item_15 = None
        silu_7 = torch.nn.functional.silu(batch_norm_7, inplace=True)
        batch_norm_7 = None
        chunk_1 = silu_7.chunk(2, 1)
        silu_7 = None
        getitem_2 = chunk_1[0]
        getitem_3 = chunk_1[1]
        chunk_1 = None
        conv2d_8 = torch.conv2d(
            getitem_3,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        item_16 = (
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_momentum = (
            None
        )
        item_17 = (
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_eps = (
            None
        )
        batch_norm_8 = torch.nn.functional.batch_norm(
            conv2d_8,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_16,
            item_17,
        )
        conv2d_8 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (item_16) = (
            item_17
        ) = None
        silu_8 = torch.nn.functional.silu(batch_norm_8, inplace=True)
        batch_norm_8 = None
        conv2d_9 = torch.conv2d(
            silu_8,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_8 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        item_18 = (
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_momentum = (
            None
        )
        item_19 = (
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_eps.item()
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_eps = (
            None
        )
        batch_norm_9 = torch.nn.functional.batch_norm(
            conv2d_9,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_18,
            item_19,
        )
        conv2d_9 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (item_18) = (
            item_19
        ) = None
        silu_9 = torch.nn.functional.silu(batch_norm_9, inplace=True)
        batch_norm_9 = None
        add_1 = getitem_3 + silu_9
        silu_9 = None
        cat_1 = torch.cat([getitem_2, getitem_3, add_1], 1)
        getitem_2 = getitem_3 = add_1 = None
        conv2d_10 = torch.conv2d(
            cat_1,
            l_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = (
            l_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_
        ) = None
        item_20 = l_self_modules_model_modules_4_modules_cv2_modules_bn_momentum.item()
        l_self_modules_model_modules_4_modules_cv2_modules_bn_momentum = None
        item_21 = l_self_modules_model_modules_4_modules_cv2_modules_bn_eps.item()
        l_self_modules_model_modules_4_modules_cv2_modules_bn_eps = None
        batch_norm_10 = torch.nn.functional.batch_norm(
            conv2d_10,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_20,
            item_21,
        )
        conv2d_10 = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_
        ) = item_20 = item_21 = None
        x_4 = torch.nn.functional.silu(batch_norm_10, inplace=True)
        batch_norm_10 = None
        conv2d_11 = torch.conv2d(
            x_4,
            l_self_modules_model_modules_5_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_model_modules_5_modules_conv_parameters_weight_ = None
        item_22 = l_self_modules_model_modules_5_modules_bn_momentum.item()
        l_self_modules_model_modules_5_modules_bn_momentum = None
        item_23 = l_self_modules_model_modules_5_modules_bn_eps.item()
        l_self_modules_model_modules_5_modules_bn_eps = None
        batch_norm_11 = torch.nn.functional.batch_norm(
            conv2d_11,
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_5_modules_bn_parameters_bias_,
            False,
            item_22,
            item_23,
        )
        conv2d_11 = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_5_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_5_modules_bn_parameters_bias_
        ) = item_22 = item_23 = None
        x_5 = torch.nn.functional.silu(batch_norm_11, inplace=True)
        batch_norm_11 = None
        conv2d_12 = torch.conv2d(
            x_5,
            l_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = (
            l_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_
        ) = None
        item_24 = l_self_modules_model_modules_6_modules_cv1_modules_bn_momentum.item()
        l_self_modules_model_modules_6_modules_cv1_modules_bn_momentum = None
        item_25 = l_self_modules_model_modules_6_modules_cv1_modules_bn_eps.item()
        l_self_modules_model_modules_6_modules_cv1_modules_bn_eps = None
        batch_norm_12 = torch.nn.functional.batch_norm(
            conv2d_12,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_24,
            item_25,
        )
        conv2d_12 = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_
        ) = item_24 = item_25 = None
        silu_12 = torch.nn.functional.silu(batch_norm_12, inplace=True)
        batch_norm_12 = None
        chunk_2 = silu_12.chunk(2, 1)
        silu_12 = None
        getitem_4 = chunk_2[0]
        getitem_5 = chunk_2[1]
        chunk_2 = None
        conv2d_13 = torch.conv2d(
            getitem_5,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        item_26 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_momentum = (
            None
        )
        item_27 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_eps = (
            None
        )
        batch_norm_13 = torch.nn.functional.batch_norm(
            conv2d_13,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_26,
            item_27,
        )
        conv2d_13 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (item_26) = (
            item_27
        ) = None
        silu_13 = torch.nn.functional.silu(batch_norm_13, inplace=True)
        batch_norm_13 = None
        conv2d_14 = torch.conv2d(
            silu_13,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        item_28 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum = (
            None
        )
        item_29 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps = (
            None
        )
        batch_norm_14 = torch.nn.functional.batch_norm(
            conv2d_14,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_28,
            item_29,
        )
        conv2d_14 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (item_28) = (
            item_29
        ) = None
        silu_14 = torch.nn.functional.silu(batch_norm_14, inplace=True)
        batch_norm_14 = None
        conv2d_15 = torch.conv2d(
            silu_14,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_14 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        item_30 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum = (
            None
        )
        item_31 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps = (
            None
        )
        batch_norm_15 = torch.nn.functional.batch_norm(
            conv2d_15,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_30,
            item_31,
        )
        conv2d_15 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (item_30) = (
            item_31
        ) = None
        silu_15 = torch.nn.functional.silu(batch_norm_15, inplace=True)
        batch_norm_15 = None
        input_1 = silu_13 + silu_15
        silu_13 = silu_15 = None
        conv2d_16 = torch.conv2d(
            input_1,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        item_32 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum = (
            None
        )
        item_33 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps = (
            None
        )
        batch_norm_16 = torch.nn.functional.batch_norm(
            conv2d_16,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_32,
            item_33,
        )
        conv2d_16 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (item_32) = (
            item_33
        ) = None
        silu_16 = torch.nn.functional.silu(batch_norm_16, inplace=True)
        batch_norm_16 = None
        conv2d_17 = torch.conv2d(
            silu_16,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_16 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        item_34 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum = (
            None
        )
        item_35 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps = (
            None
        )
        batch_norm_17 = torch.nn.functional.batch_norm(
            conv2d_17,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_34,
            item_35,
        )
        conv2d_17 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (item_34) = (
            item_35
        ) = None
        silu_17 = torch.nn.functional.silu(batch_norm_17, inplace=True)
        batch_norm_17 = None
        input_2 = input_1 + silu_17
        input_1 = silu_17 = None
        conv2d_18 = torch.conv2d(
            getitem_5,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        item_36 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_momentum = (
            None
        )
        item_37 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_eps = (
            None
        )
        batch_norm_18 = torch.nn.functional.batch_norm(
            conv2d_18,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_36,
            item_37,
        )
        conv2d_18 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (item_36) = (
            item_37
        ) = None
        silu_18 = torch.nn.functional.silu(batch_norm_18, inplace=True)
        batch_norm_18 = None
        cat_2 = torch.cat((input_2, silu_18), 1)
        input_2 = silu_18 = None
        conv2d_19 = torch.conv2d(
            cat_2,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        item_38 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_momentum = (
            None
        )
        item_39 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_eps = (
            None
        )
        batch_norm_19 = torch.nn.functional.batch_norm(
            conv2d_19,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            item_38,
            item_39,
        )
        conv2d_19 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (item_38) = (
            item_39
        ) = None
        silu_19 = torch.nn.functional.silu(batch_norm_19, inplace=True)
        batch_norm_19 = None
        cat_3 = torch.cat([getitem_4, getitem_5, silu_19], 1)
        getitem_4 = getitem_5 = silu_19 = None
        conv2d_20 = torch.conv2d(
            cat_3,
            l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = (
            l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_
        ) = None
        item_40 = l_self_modules_model_modules_6_modules_cv2_modules_bn_momentum.item()
        l_self_modules_model_modules_6_modules_cv2_modules_bn_momentum = None
        item_41 = l_self_modules_model_modules_6_modules_cv2_modules_bn_eps.item()
        l_self_modules_model_modules_6_modules_cv2_modules_bn_eps = None
        batch_norm_20 = torch.nn.functional.batch_norm(
            conv2d_20,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_40,
            item_41,
        )
        conv2d_20 = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_
        ) = item_40 = item_41 = None
        x_6 = torch.nn.functional.silu(batch_norm_20, inplace=True)
        batch_norm_20 = None
        conv2d_21 = torch.conv2d(
            x_6,
            l_self_modules_model_modules_7_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_model_modules_7_modules_conv_parameters_weight_ = None
        item_42 = l_self_modules_model_modules_7_modules_bn_momentum.item()
        l_self_modules_model_modules_7_modules_bn_momentum = None
        item_43 = l_self_modules_model_modules_7_modules_bn_eps.item()
        l_self_modules_model_modules_7_modules_bn_eps = None
        batch_norm_21 = torch.nn.functional.batch_norm(
            conv2d_21,
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_7_modules_bn_parameters_weight_,
            l_self_modules_model_modules_7_modules_bn_parameters_bias_,
            False,
            item_42,
            item_43,
        )
        conv2d_21 = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_7_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_7_modules_bn_parameters_bias_
        ) = item_42 = item_43 = None
        x_7 = torch.nn.functional.silu(batch_norm_21, inplace=True)
        batch_norm_21 = None
        conv2d_22 = torch.conv2d(
            x_7,
            l_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = (
            l_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_
        ) = None
        item_44 = l_self_modules_model_modules_8_modules_cv1_modules_bn_momentum.item()
        l_self_modules_model_modules_8_modules_cv1_modules_bn_momentum = None
        item_45 = l_self_modules_model_modules_8_modules_cv1_modules_bn_eps.item()
        l_self_modules_model_modules_8_modules_cv1_modules_bn_eps = None
        batch_norm_22 = torch.nn.functional.batch_norm(
            conv2d_22,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_44,
            item_45,
        )
        conv2d_22 = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_
        ) = item_44 = item_45 = None
        silu_22 = torch.nn.functional.silu(batch_norm_22, inplace=True)
        batch_norm_22 = None
        chunk_3 = silu_22.chunk(2, 1)
        silu_22 = None
        getitem_6 = chunk_3[0]
        getitem_7 = chunk_3[1]
        chunk_3 = None
        conv2d_23 = torch.conv2d(
            getitem_7,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        item_46 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_momentum = (
            None
        )
        item_47 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_eps = (
            None
        )
        batch_norm_23 = torch.nn.functional.batch_norm(
            conv2d_23,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_46,
            item_47,
        )
        conv2d_23 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (item_46) = (
            item_47
        ) = None
        silu_23 = torch.nn.functional.silu(batch_norm_23, inplace=True)
        batch_norm_23 = None
        conv2d_24 = torch.conv2d(
            silu_23,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        item_48 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum = (
            None
        )
        item_49 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps = (
            None
        )
        batch_norm_24 = torch.nn.functional.batch_norm(
            conv2d_24,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_48,
            item_49,
        )
        conv2d_24 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (item_48) = (
            item_49
        ) = None
        silu_24 = torch.nn.functional.silu(batch_norm_24, inplace=True)
        batch_norm_24 = None
        conv2d_25 = torch.conv2d(
            silu_24,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_24 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        item_50 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum = (
            None
        )
        item_51 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps = (
            None
        )
        batch_norm_25 = torch.nn.functional.batch_norm(
            conv2d_25,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_50,
            item_51,
        )
        conv2d_25 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (item_50) = (
            item_51
        ) = None
        silu_25 = torch.nn.functional.silu(batch_norm_25, inplace=True)
        batch_norm_25 = None
        input_3 = silu_23 + silu_25
        silu_23 = silu_25 = None
        conv2d_26 = torch.conv2d(
            input_3,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        item_52 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum = (
            None
        )
        item_53 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps = (
            None
        )
        batch_norm_26 = torch.nn.functional.batch_norm(
            conv2d_26,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_52,
            item_53,
        )
        conv2d_26 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (item_52) = (
            item_53
        ) = None
        silu_26 = torch.nn.functional.silu(batch_norm_26, inplace=True)
        batch_norm_26 = None
        conv2d_27 = torch.conv2d(
            silu_26,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_26 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        item_54 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum = (
            None
        )
        item_55 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps = (
            None
        )
        batch_norm_27 = torch.nn.functional.batch_norm(
            conv2d_27,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_54,
            item_55,
        )
        conv2d_27 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (item_54) = (
            item_55
        ) = None
        silu_27 = torch.nn.functional.silu(batch_norm_27, inplace=True)
        batch_norm_27 = None
        input_4 = input_3 + silu_27
        input_3 = silu_27 = None
        conv2d_28 = torch.conv2d(
            getitem_7,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        item_56 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_momentum = (
            None
        )
        item_57 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_eps = (
            None
        )
        batch_norm_28 = torch.nn.functional.batch_norm(
            conv2d_28,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_56,
            item_57,
        )
        conv2d_28 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (item_56) = (
            item_57
        ) = None
        silu_28 = torch.nn.functional.silu(batch_norm_28, inplace=True)
        batch_norm_28 = None
        cat_4 = torch.cat((input_4, silu_28), 1)
        input_4 = silu_28 = None
        conv2d_29 = torch.conv2d(
            cat_4,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_4 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        item_58 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_momentum = (
            None
        )
        item_59 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_eps = (
            None
        )
        batch_norm_29 = torch.nn.functional.batch_norm(
            conv2d_29,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            item_58,
            item_59,
        )
        conv2d_29 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (item_58) = (
            item_59
        ) = None
        silu_29 = torch.nn.functional.silu(batch_norm_29, inplace=True)
        batch_norm_29 = None
        cat_5 = torch.cat([getitem_6, getitem_7, silu_29], 1)
        getitem_6 = getitem_7 = silu_29 = None
        conv2d_30 = torch.conv2d(
            cat_5,
            l_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_5 = (
            l_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_
        ) = None
        item_60 = l_self_modules_model_modules_8_modules_cv2_modules_bn_momentum.item()
        l_self_modules_model_modules_8_modules_cv2_modules_bn_momentum = None
        item_61 = l_self_modules_model_modules_8_modules_cv2_modules_bn_eps.item()
        l_self_modules_model_modules_8_modules_cv2_modules_bn_eps = None
        batch_norm_30 = torch.nn.functional.batch_norm(
            conv2d_30,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_60,
            item_61,
        )
        conv2d_30 = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_
        ) = item_60 = item_61 = None
        x_8 = torch.nn.functional.silu(batch_norm_30, inplace=True)
        batch_norm_30 = None
        conv2d_31 = torch.conv2d(
            x_8,
            l_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = (
            l_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_
        ) = None
        item_62 = l_self_modules_model_modules_9_modules_cv1_modules_bn_momentum.item()
        l_self_modules_model_modules_9_modules_cv1_modules_bn_momentum = None
        item_63 = l_self_modules_model_modules_9_modules_cv1_modules_bn_eps.item()
        l_self_modules_model_modules_9_modules_cv1_modules_bn_eps = None
        batch_norm_31 = torch.nn.functional.batch_norm(
            conv2d_31,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_62,
            item_63,
        )
        conv2d_31 = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_
        ) = item_62 = item_63 = None
        silu_31 = torch.nn.functional.silu(batch_norm_31, inplace=True)
        batch_norm_31 = None
        split = silu_31.split((256, 256), dim=1)
        silu_31 = None
        a = split[0]
        b = split[1]
        split = None
        conv2d_32 = torch.conv2d(
            b,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        qkv = torch.nn.functional.batch_norm(
            conv2d_32,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_32 = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        view = qkv.view(1, 4, 128, 400)
        qkv = None
        split_1 = view.split([32, 32, 64], dim=2)
        view = None
        q = split_1[0]
        k = split_1[1]
        v = split_1[2]
        split_1 = None
        transpose = q.transpose(-2, -1)
        q = None
        matmul = transpose @ k
        transpose = k = None
        attn = matmul * 0.1767766952966369
        matmul = None
        attn_1 = attn.softmax(dim=-1)
        attn = None
        transpose_1 = attn_1.transpose(-2, -1)
        attn_1 = None
        matmul_1 = v @ transpose_1
        transpose_1 = None
        view_1 = matmul_1.view(1, 256, 20, 20)
        matmul_1 = None
        reshape = v.reshape(1, 256, 20, 20)
        v = None
        conv2d_33 = torch.conv2d(
            reshape,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        reshape = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_33 = torch.nn.functional.batch_norm(
            conv2d_33,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_33 = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_9 = view_1 + batch_norm_33
        view_1 = batch_norm_33 = None
        conv2d_34 = torch.conv2d(
            x_9,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            conv2d_34,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_34 = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_11 = b + x_10
        b = x_10 = None
        conv2d_35 = torch.conv2d(
            x_11,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_35 = torch.nn.functional.batch_norm(
            conv2d_35,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_35 = l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_ = (None)
        input_5 = torch.nn.functional.silu(batch_norm_35, inplace=True)
        batch_norm_35 = None
        conv2d_36 = torch.conv2d(
            input_5,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_ = (None)
        input_6 = torch.nn.functional.batch_norm(
            conv2d_36,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_36 = l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_ = (None)
        x_12 = x_11 + input_6
        x_11 = input_6 = None
        cat_6 = torch.cat((a, x_12), 1)
        a = x_12 = None
        conv2d_37 = torch.conv2d(
            cat_6,
            l_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_6 = (
            l_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_
        ) = None
        item_64 = l_self_modules_model_modules_9_modules_cv2_modules_bn_momentum.item()
        l_self_modules_model_modules_9_modules_cv2_modules_bn_momentum = None
        item_65 = l_self_modules_model_modules_9_modules_cv2_modules_bn_eps.item()
        l_self_modules_model_modules_9_modules_cv2_modules_bn_eps = None
        batch_norm_37 = torch.nn.functional.batch_norm(
            conv2d_37,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_64,
            item_65,
        )
        conv2d_37 = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_
        ) = item_64 = item_65 = None
        x_13 = torch.nn.functional.silu(batch_norm_37, inplace=True)
        batch_norm_37 = None
        conv2d_38 = torch.conv2d(
            x_13,
            l_self_modules_model_modules_10_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = (
            l_self_modules_model_modules_10_modules_conv_modules_conv_parameters_weight_
        ) = None
        batch_norm_38 = torch.nn.functional.batch_norm(
            conv2d_38,
            l_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_38 = l_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_mean_ = (
            l_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_10_modules_conv_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_10_modules_conv_modules_bn_parameters_bias_
        ) = None
        silu_34 = torch.nn.functional.silu(batch_norm_38, inplace=True)
        batch_norm_38 = None
        adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d(silu_34, 1)
        silu_34 = None
        flatten = adaptive_avg_pool2d.flatten(1)
        adaptive_avg_pool2d = None
        dropout = torch.nn.functional.dropout(flatten, 0.0, False, True)
        flatten = None
        x_14 = torch._C._nn.linear(
            dropout,
            l_self_modules_model_modules_10_modules_linear_parameters_weight_,
            l_self_modules_model_modules_10_modules_linear_parameters_bias_,
        )
        dropout = (
            l_self_modules_model_modules_10_modules_linear_parameters_weight_
        ) = l_self_modules_model_modules_10_modules_linear_parameters_bias_ = None
        y = x_14.softmax(1)
        return (y, x_14)
