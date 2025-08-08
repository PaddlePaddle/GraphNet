import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_model_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_7_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_7_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_linear_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_linear_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_model_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_0_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_1_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_3_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_5_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_5_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_6_parameters_gamma_ = (
            L_self_modules_model_modules_6_parameters_gamma_
        )
        l_self_modules_model_modules_7_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_7_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_8_parameters_gamma_ = (
            L_self_modules_model_modules_8_parameters_gamma_
        )
        l_self_modules_model_modules_9_modules_conv_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_conv_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_conv_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_9_modules_conv_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_9_modules_linear_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_linear_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_linear_parameters_bias_ = (
            L_self_modules_model_modules_9_modules_linear_parameters_bias_
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
        batch_norm = torch.nn.functional.batch_norm(
            conv2d,
            l_self_modules_model_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d = (
            l_self_modules_model_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_0_modules_bn_parameters_bias_ = None
        x = torch.nn.functional.silu(batch_norm, inplace=False)
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
        batch_norm_1 = torch.nn.functional.batch_norm(
            conv2d_1,
            l_self_modules_model_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_1 = (
            l_self_modules_model_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_1_modules_bn_parameters_bias_ = None
        x_1 = torch.nn.functional.silu(batch_norm_1, inplace=False)
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
        batch_norm_2 = torch.nn.functional.batch_norm(
            conv2d_2,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_2 = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_2 = torch.nn.functional.silu(batch_norm_2, inplace=False)
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
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_3 = torch.nn.functional.batch_norm(
            conv2d_3,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_3 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_3 = torch.nn.functional.silu(batch_norm_3, inplace=False)
        batch_norm_3 = None
        conv2d_4 = torch.conv2d(
            silu_3,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_4 = torch.nn.functional.batch_norm(
            conv2d_4,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_4 = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_4 = torch.nn.functional.silu(batch_norm_4, inplace=False)
        batch_norm_4 = None
        conv2d_5 = torch.conv2d(
            silu_4,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_4 = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_5 = torch.nn.functional.batch_norm(
            conv2d_5,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_5 = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_5 = torch.nn.functional.silu(batch_norm_5, inplace=False)
        batch_norm_5 = None
        input_1 = silu_3 + silu_5
        silu_3 = silu_5 = None
        conv2d_6 = torch.conv2d(
            input_1,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_6 = torch.nn.functional.batch_norm(
            conv2d_6,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_6 = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_6 = torch.nn.functional.silu(batch_norm_6, inplace=False)
        batch_norm_6 = None
        conv2d_7 = torch.conv2d(
            silu_6,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_6 = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_7 = torch.nn.functional.batch_norm(
            conv2d_7,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_7 = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_7 = torch.nn.functional.silu(batch_norm_7, inplace=False)
        batch_norm_7 = None
        input_2 = input_1 + silu_7
        input_1 = silu_7 = None
        conv2d_8 = torch.conv2d(
            getitem_1,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_8 = torch.nn.functional.batch_norm(
            conv2d_8,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_8 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_8 = torch.nn.functional.silu(batch_norm_8, inplace=False)
        batch_norm_8 = None
        cat = torch.cat((input_2, silu_8), 1)
        input_2 = silu_8 = None
        conv2d_9 = torch.conv2d(
            cat,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_9 = torch.nn.functional.batch_norm(
            conv2d_9,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_9 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_9 = torch.nn.functional.silu(batch_norm_9, inplace=False)
        batch_norm_9 = None
        conv2d_10 = torch.conv2d(
            getitem_1,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_10 = torch.nn.functional.batch_norm(
            conv2d_10,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_10 = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_10 = torch.nn.functional.silu(batch_norm_10, inplace=False)
        batch_norm_10 = None
        conv2d_11 = torch.conv2d(
            silu_10,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_11 = torch.nn.functional.batch_norm(
            conv2d_11,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_11 = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_11 = torch.nn.functional.silu(batch_norm_11, inplace=False)
        batch_norm_11 = None
        conv2d_12 = torch.conv2d(
            silu_11,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_11 = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_12 = torch.nn.functional.batch_norm(
            conv2d_12,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_12 = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_12 = torch.nn.functional.silu(batch_norm_12, inplace=False)
        batch_norm_12 = None
        input_3 = silu_10 + silu_12
        silu_10 = silu_12 = None
        conv2d_13 = torch.conv2d(
            input_3,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_13 = torch.nn.functional.batch_norm(
            conv2d_13,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_13 = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_13 = torch.nn.functional.silu(batch_norm_13, inplace=False)
        batch_norm_13 = None
        conv2d_14 = torch.conv2d(
            silu_13,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_13 = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_14 = torch.nn.functional.batch_norm(
            conv2d_14,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_14 = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_14 = torch.nn.functional.silu(batch_norm_14, inplace=False)
        batch_norm_14 = None
        input_4 = input_3 + silu_14
        input_3 = silu_14 = None
        conv2d_15 = torch.conv2d(
            getitem_1,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_15 = torch.nn.functional.batch_norm(
            conv2d_15,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_15 = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_15 = torch.nn.functional.silu(batch_norm_15, inplace=False)
        batch_norm_15 = None
        cat_1 = torch.cat((input_4, silu_15), 1)
        input_4 = silu_15 = None
        conv2d_16 = torch.conv2d(
            cat_1,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_16 = torch.nn.functional.batch_norm(
            conv2d_16,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_16 = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_16 = torch.nn.functional.silu(batch_norm_16, inplace=False)
        batch_norm_16 = None
        cat_2 = torch.cat([getitem, getitem_1, silu_9, silu_16], 1)
        getitem = getitem_1 = silu_9 = silu_16 = None
        conv2d_17 = torch.conv2d(
            cat_2,
            l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = (
            l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_17 = torch.nn.functional.batch_norm(
            conv2d_17,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_17 = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_2 = torch.nn.functional.silu(batch_norm_17, inplace=False)
        batch_norm_17 = None
        conv2d_18 = torch.conv2d(
            x_2,
            l_self_modules_model_modules_3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_model_modules_3_modules_conv_parameters_weight_ = None
        batch_norm_18 = torch.nn.functional.batch_norm(
            conv2d_18,
            l_self_modules_model_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_18 = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_3_modules_bn_parameters_bias_ = None
        x_3 = torch.nn.functional.silu(batch_norm_18, inplace=False)
        batch_norm_18 = None
        conv2d_19 = torch.conv2d(
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
        batch_norm_19 = torch.nn.functional.batch_norm(
            conv2d_19,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_19 = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_19 = torch.nn.functional.silu(batch_norm_19, inplace=False)
        batch_norm_19 = None
        chunk_1 = silu_19.chunk(2, 1)
        silu_19 = None
        getitem_2 = chunk_1[0]
        getitem_3 = chunk_1[1]
        chunk_1 = None
        conv2d_20 = torch.conv2d(
            getitem_3,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_20 = torch.nn.functional.batch_norm(
            conv2d_20,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_20 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_20 = torch.nn.functional.silu(batch_norm_20, inplace=False)
        batch_norm_20 = None
        conv2d_21 = torch.conv2d(
            silu_20,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_21 = torch.nn.functional.batch_norm(
            conv2d_21,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_21 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_21 = torch.nn.functional.silu(batch_norm_21, inplace=False)
        batch_norm_21 = None
        conv2d_22 = torch.conv2d(
            silu_21,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_21 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_22 = torch.nn.functional.batch_norm(
            conv2d_22,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_22 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_22 = torch.nn.functional.silu(batch_norm_22, inplace=False)
        batch_norm_22 = None
        input_5 = silu_20 + silu_22
        silu_20 = silu_22 = None
        conv2d_23 = torch.conv2d(
            input_5,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_23 = torch.nn.functional.batch_norm(
            conv2d_23,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_23 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_23 = torch.nn.functional.silu(batch_norm_23, inplace=False)
        batch_norm_23 = None
        conv2d_24 = torch.conv2d(
            silu_23,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_23 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_24 = torch.nn.functional.batch_norm(
            conv2d_24,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_24 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_24 = torch.nn.functional.silu(batch_norm_24, inplace=False)
        batch_norm_24 = None
        input_6 = input_5 + silu_24
        input_5 = silu_24 = None
        conv2d_25 = torch.conv2d(
            getitem_3,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_25 = torch.nn.functional.batch_norm(
            conv2d_25,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_25 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_25 = torch.nn.functional.silu(batch_norm_25, inplace=False)
        batch_norm_25 = None
        cat_3 = torch.cat((input_6, silu_25), 1)
        input_6 = silu_25 = None
        conv2d_26 = torch.conv2d(
            cat_3,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_26 = torch.nn.functional.batch_norm(
            conv2d_26,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_26 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_26 = torch.nn.functional.silu(batch_norm_26, inplace=False)
        batch_norm_26 = None
        conv2d_27 = torch.conv2d(
            getitem_3,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_27 = torch.nn.functional.batch_norm(
            conv2d_27,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_27 = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_27 = torch.nn.functional.silu(batch_norm_27, inplace=False)
        batch_norm_27 = None
        conv2d_28 = torch.conv2d(
            silu_27,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_28 = torch.nn.functional.batch_norm(
            conv2d_28,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_28 = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_28 = torch.nn.functional.silu(batch_norm_28, inplace=False)
        batch_norm_28 = None
        conv2d_29 = torch.conv2d(
            silu_28,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_28 = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_29 = torch.nn.functional.batch_norm(
            conv2d_29,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_29 = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_29 = torch.nn.functional.silu(batch_norm_29, inplace=False)
        batch_norm_29 = None
        input_7 = silu_27 + silu_29
        silu_27 = silu_29 = None
        conv2d_30 = torch.conv2d(
            input_7,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_30 = torch.nn.functional.batch_norm(
            conv2d_30,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_30 = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_30 = torch.nn.functional.silu(batch_norm_30, inplace=False)
        batch_norm_30 = None
        conv2d_31 = torch.conv2d(
            silu_30,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_30 = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_31 = torch.nn.functional.batch_norm(
            conv2d_31,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_31 = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_31 = torch.nn.functional.silu(batch_norm_31, inplace=False)
        batch_norm_31 = None
        input_8 = input_7 + silu_31
        input_7 = silu_31 = None
        conv2d_32 = torch.conv2d(
            getitem_3,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_32 = torch.nn.functional.batch_norm(
            conv2d_32,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_32 = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_32 = torch.nn.functional.silu(batch_norm_32, inplace=False)
        batch_norm_32 = None
        cat_4 = torch.cat((input_8, silu_32), 1)
        input_8 = silu_32 = None
        conv2d_33 = torch.conv2d(
            cat_4,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_4 = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_33 = torch.nn.functional.batch_norm(
            conv2d_33,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_33 = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_33 = torch.nn.functional.silu(batch_norm_33, inplace=False)
        batch_norm_33 = None
        cat_5 = torch.cat([getitem_2, getitem_3, silu_26, silu_33], 1)
        getitem_2 = getitem_3 = silu_26 = silu_33 = None
        conv2d_34 = torch.conv2d(
            cat_5,
            l_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_5 = (
            l_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_34 = torch.nn.functional.batch_norm(
            conv2d_34,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_34 = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_4 = torch.nn.functional.silu(batch_norm_34, inplace=False)
        batch_norm_34 = None
        conv2d_35 = torch.conv2d(
            x_4,
            l_self_modules_model_modules_5_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_model_modules_5_modules_conv_parameters_weight_ = None
        batch_norm_35 = torch.nn.functional.batch_norm(
            conv2d_35,
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_35 = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_5_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_5_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.silu(batch_norm_35, inplace=False)
        batch_norm_35 = None
        conv2d_36 = torch.conv2d(
            x_5,
            l_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_36 = torch.nn.functional.batch_norm(
            conv2d_36,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_36 = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_36 = torch.nn.functional.silu(batch_norm_36, inplace=False)
        batch_norm_36 = None
        conv2d_37 = torch.conv2d(
            silu_36,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_37 = torch.nn.functional.batch_norm(
            conv2d_37,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_37 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten = batch_norm_37.flatten(2)
        batch_norm_37 = None
        qkv = flatten.transpose(1, 2)
        flatten = None
        qkv_1 = qkv.reshape(4, 400, 768)
        qkv = None
        view = qkv_1.view(4, 400, 8, 96)
        qkv_1 = None
        permute = view.permute(0, 2, 3, 1)
        view = None
        split = permute.split([32, 32, 32], dim=2)
        permute = None
        q = split[0]
        k = split[1]
        v = split[2]
        split = None
        transpose_1 = q.transpose(-2, -1)
        q = None
        matmul = transpose_1 @ k
        transpose_1 = k = None
        attn = matmul * 0.1767766952966369
        matmul = None
        attn_1 = attn.softmax(dim=-1)
        attn = None
        transpose_2 = attn_1.transpose(-2, -1)
        attn_1 = None
        x_6 = v @ transpose_2
        transpose_2 = None
        x_7 = x_6.permute(0, 3, 1, 2)
        x_6 = None
        v_1 = v.permute(0, 3, 1, 2)
        v = None
        x_8 = x_7.reshape(1, 1600, 256)
        x_7 = None
        v_2 = v_1.reshape(1, 1600, 256)
        v_1 = None
        reshape_3 = x_8.reshape(1, 40, 40, 256)
        x_8 = None
        permute_3 = reshape_3.permute(0, 3, 1, 2)
        reshape_3 = None
        x_9 = permute_3.contiguous()
        permute_3 = None
        reshape_4 = v_2.reshape(1, 40, 40, 256)
        v_2 = None
        permute_4 = reshape_4.permute(0, 3, 1, 2)
        reshape_4 = None
        v_3 = permute_4.contiguous()
        permute_4 = None
        conv2d_38 = torch.conv2d(
            v_3,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_3 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_38 = torch.nn.functional.batch_norm(
            conv2d_38,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_38 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_10 = x_9 + batch_norm_38
        x_9 = batch_norm_38 = None
        conv2d_39 = torch.conv2d(
            x_10,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_39 = torch.nn.functional.batch_norm(
            conv2d_39,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_39 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_11 = silu_36 + batch_norm_39
        batch_norm_39 = None
        conv2d_40 = torch.conv2d(
            x_11,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_40 = torch.nn.functional.batch_norm(
            conv2d_40,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_40 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_9 = torch.nn.functional.silu(batch_norm_40, inplace=False)
        batch_norm_40 = None
        conv2d_41 = torch.conv2d(
            input_9,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_9 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_10 = torch.nn.functional.batch_norm(
            conv2d_41,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_41 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_11 = x_11 + input_10
        x_11 = input_10 = None
        conv2d_42 = torch.conv2d(
            input_11,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_42 = torch.nn.functional.batch_norm(
            conv2d_42,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_42 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_1 = batch_norm_42.flatten(2)
        batch_norm_42 = None
        qkv_2 = flatten_1.transpose(1, 2)
        flatten_1 = None
        qkv_3 = qkv_2.reshape(4, 400, 768)
        qkv_2 = None
        view_1 = qkv_3.view(4, 400, 8, 96)
        qkv_3 = None
        permute_5 = view_1.permute(0, 2, 3, 1)
        view_1 = None
        split_1 = permute_5.split([32, 32, 32], dim=2)
        permute_5 = None
        q_1 = split_1[0]
        k_1 = split_1[1]
        v_4 = split_1[2]
        split_1 = None
        transpose_4 = q_1.transpose(-2, -1)
        q_1 = None
        matmul_2 = transpose_4 @ k_1
        transpose_4 = k_1 = None
        attn_2 = matmul_2 * 0.1767766952966369
        matmul_2 = None
        attn_3 = attn_2.softmax(dim=-1)
        attn_2 = None
        transpose_5 = attn_3.transpose(-2, -1)
        attn_3 = None
        x_12 = v_4 @ transpose_5
        transpose_5 = None
        x_13 = x_12.permute(0, 3, 1, 2)
        x_12 = None
        v_5 = v_4.permute(0, 3, 1, 2)
        v_4 = None
        x_14 = x_13.reshape(1, 1600, 256)
        x_13 = None
        v_6 = v_5.reshape(1, 1600, 256)
        v_5 = None
        reshape_8 = x_14.reshape(1, 40, 40, 256)
        x_14 = None
        permute_8 = reshape_8.permute(0, 3, 1, 2)
        reshape_8 = None
        x_15 = permute_8.contiguous()
        permute_8 = None
        reshape_9 = v_6.reshape(1, 40, 40, 256)
        v_6 = None
        permute_9 = reshape_9.permute(0, 3, 1, 2)
        reshape_9 = None
        v_7 = permute_9.contiguous()
        permute_9 = None
        conv2d_43 = torch.conv2d(
            v_7,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_7 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_43 = torch.nn.functional.batch_norm(
            conv2d_43,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_43 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_16 = x_15 + batch_norm_43
        x_15 = batch_norm_43 = None
        conv2d_44 = torch.conv2d(
            x_16,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_44 = torch.nn.functional.batch_norm(
            conv2d_44,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_44 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_17 = input_11 + batch_norm_44
        input_11 = batch_norm_44 = None
        conv2d_45 = torch.conv2d(
            x_17,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_45 = torch.nn.functional.batch_norm(
            conv2d_45,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_45 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_12 = torch.nn.functional.silu(batch_norm_45, inplace=False)
        batch_norm_45 = None
        conv2d_46 = torch.conv2d(
            input_12,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_12 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            conv2d_46,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_46 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_14 = x_17 + input_13
        x_17 = input_13 = None
        conv2d_47 = torch.conv2d(
            silu_36,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_47 = torch.nn.functional.batch_norm(
            conv2d_47,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_47 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_2 = batch_norm_47.flatten(2)
        batch_norm_47 = None
        qkv_4 = flatten_2.transpose(1, 2)
        flatten_2 = None
        qkv_5 = qkv_4.reshape(4, 400, 768)
        qkv_4 = None
        view_2 = qkv_5.view(4, 400, 8, 96)
        qkv_5 = None
        permute_10 = view_2.permute(0, 2, 3, 1)
        view_2 = None
        split_2 = permute_10.split([32, 32, 32], dim=2)
        permute_10 = None
        q_2 = split_2[0]
        k_2 = split_2[1]
        v_8 = split_2[2]
        split_2 = None
        transpose_7 = q_2.transpose(-2, -1)
        q_2 = None
        matmul_4 = transpose_7 @ k_2
        transpose_7 = k_2 = None
        attn_4 = matmul_4 * 0.1767766952966369
        matmul_4 = None
        attn_5 = attn_4.softmax(dim=-1)
        attn_4 = None
        transpose_8 = attn_5.transpose(-2, -1)
        attn_5 = None
        x_18 = v_8 @ transpose_8
        transpose_8 = None
        x_19 = x_18.permute(0, 3, 1, 2)
        x_18 = None
        v_9 = v_8.permute(0, 3, 1, 2)
        v_8 = None
        x_20 = x_19.reshape(1, 1600, 256)
        x_19 = None
        v_10 = v_9.reshape(1, 1600, 256)
        v_9 = None
        reshape_13 = x_20.reshape(1, 40, 40, 256)
        x_20 = None
        permute_13 = reshape_13.permute(0, 3, 1, 2)
        reshape_13 = None
        x_21 = permute_13.contiguous()
        permute_13 = None
        reshape_14 = v_10.reshape(1, 40, 40, 256)
        v_10 = None
        permute_14 = reshape_14.permute(0, 3, 1, 2)
        reshape_14 = None
        v_11 = permute_14.contiguous()
        permute_14 = None
        conv2d_48 = torch.conv2d(
            v_11,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_11 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_48 = torch.nn.functional.batch_norm(
            conv2d_48,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_48 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_22 = x_21 + batch_norm_48
        x_21 = batch_norm_48 = None
        conv2d_49 = torch.conv2d(
            x_22,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_49 = torch.nn.functional.batch_norm(
            conv2d_49,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_49 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_23 = silu_36 + batch_norm_49
        batch_norm_49 = None
        conv2d_50 = torch.conv2d(
            x_23,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_50 = torch.nn.functional.batch_norm(
            conv2d_50,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_50 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_15 = torch.nn.functional.silu(batch_norm_50, inplace=False)
        batch_norm_50 = None
        conv2d_51 = torch.conv2d(
            input_15,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_15 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_16 = torch.nn.functional.batch_norm(
            conv2d_51,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_51 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_17 = x_23 + input_16
        x_23 = input_16 = None
        conv2d_52 = torch.conv2d(
            input_17,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_52 = torch.nn.functional.batch_norm(
            conv2d_52,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_52 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_3 = batch_norm_52.flatten(2)
        batch_norm_52 = None
        qkv_6 = flatten_3.transpose(1, 2)
        flatten_3 = None
        qkv_7 = qkv_6.reshape(4, 400, 768)
        qkv_6 = None
        view_3 = qkv_7.view(4, 400, 8, 96)
        qkv_7 = None
        permute_15 = view_3.permute(0, 2, 3, 1)
        view_3 = None
        split_3 = permute_15.split([32, 32, 32], dim=2)
        permute_15 = None
        q_3 = split_3[0]
        k_3 = split_3[1]
        v_12 = split_3[2]
        split_3 = None
        transpose_10 = q_3.transpose(-2, -1)
        q_3 = None
        matmul_6 = transpose_10 @ k_3
        transpose_10 = k_3 = None
        attn_6 = matmul_6 * 0.1767766952966369
        matmul_6 = None
        attn_7 = attn_6.softmax(dim=-1)
        attn_6 = None
        transpose_11 = attn_7.transpose(-2, -1)
        attn_7 = None
        x_24 = v_12 @ transpose_11
        transpose_11 = None
        x_25 = x_24.permute(0, 3, 1, 2)
        x_24 = None
        v_13 = v_12.permute(0, 3, 1, 2)
        v_12 = None
        x_26 = x_25.reshape(1, 1600, 256)
        x_25 = None
        v_14 = v_13.reshape(1, 1600, 256)
        v_13 = None
        reshape_18 = x_26.reshape(1, 40, 40, 256)
        x_26 = None
        permute_18 = reshape_18.permute(0, 3, 1, 2)
        reshape_18 = None
        x_27 = permute_18.contiguous()
        permute_18 = None
        reshape_19 = v_14.reshape(1, 40, 40, 256)
        v_14 = None
        permute_19 = reshape_19.permute(0, 3, 1, 2)
        reshape_19 = None
        v_15 = permute_19.contiguous()
        permute_19 = None
        conv2d_53 = torch.conv2d(
            v_15,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_15 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_53 = torch.nn.functional.batch_norm(
            conv2d_53,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_53 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_28 = x_27 + batch_norm_53
        x_27 = batch_norm_53 = None
        conv2d_54 = torch.conv2d(
            x_28,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_54 = torch.nn.functional.batch_norm(
            conv2d_54,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_54 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_29 = input_17 + batch_norm_54
        input_17 = batch_norm_54 = None
        conv2d_55 = torch.conv2d(
            x_29,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_55 = torch.nn.functional.batch_norm(
            conv2d_55,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_55 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_18 = torch.nn.functional.silu(batch_norm_55, inplace=False)
        batch_norm_55 = None
        conv2d_56 = torch.conv2d(
            input_18,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_18 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_19 = torch.nn.functional.batch_norm(
            conv2d_56,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_56 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_20 = x_29 + input_19
        x_29 = input_19 = None
        conv2d_57 = torch.conv2d(
            silu_36,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_57 = torch.nn.functional.batch_norm(
            conv2d_57,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_57 = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_4 = batch_norm_57.flatten(2)
        batch_norm_57 = None
        qkv_8 = flatten_4.transpose(1, 2)
        flatten_4 = None
        qkv_9 = qkv_8.reshape(4, 400, 768)
        qkv_8 = None
        view_4 = qkv_9.view(4, 400, 8, 96)
        qkv_9 = None
        permute_20 = view_4.permute(0, 2, 3, 1)
        view_4 = None
        split_4 = permute_20.split([32, 32, 32], dim=2)
        permute_20 = None
        q_4 = split_4[0]
        k_4 = split_4[1]
        v_16 = split_4[2]
        split_4 = None
        transpose_13 = q_4.transpose(-2, -1)
        q_4 = None
        matmul_8 = transpose_13 @ k_4
        transpose_13 = k_4 = None
        attn_8 = matmul_8 * 0.1767766952966369
        matmul_8 = None
        attn_9 = attn_8.softmax(dim=-1)
        attn_8 = None
        transpose_14 = attn_9.transpose(-2, -1)
        attn_9 = None
        x_30 = v_16 @ transpose_14
        transpose_14 = None
        x_31 = x_30.permute(0, 3, 1, 2)
        x_30 = None
        v_17 = v_16.permute(0, 3, 1, 2)
        v_16 = None
        x_32 = x_31.reshape(1, 1600, 256)
        x_31 = None
        v_18 = v_17.reshape(1, 1600, 256)
        v_17 = None
        reshape_23 = x_32.reshape(1, 40, 40, 256)
        x_32 = None
        permute_23 = reshape_23.permute(0, 3, 1, 2)
        reshape_23 = None
        x_33 = permute_23.contiguous()
        permute_23 = None
        reshape_24 = v_18.reshape(1, 40, 40, 256)
        v_18 = None
        permute_24 = reshape_24.permute(0, 3, 1, 2)
        reshape_24 = None
        v_19 = permute_24.contiguous()
        permute_24 = None
        conv2d_58 = torch.conv2d(
            v_19,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_19 = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_58 = torch.nn.functional.batch_norm(
            conv2d_58,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_58 = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_34 = x_33 + batch_norm_58
        x_33 = batch_norm_58 = None
        conv2d_59 = torch.conv2d(
            x_34,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_59 = torch.nn.functional.batch_norm(
            conv2d_59,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_59 = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_35 = silu_36 + batch_norm_59
        batch_norm_59 = None
        conv2d_60 = torch.conv2d(
            x_35,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_60 = torch.nn.functional.batch_norm(
            conv2d_60,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_60 = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_21 = torch.nn.functional.silu(batch_norm_60, inplace=False)
        batch_norm_60 = None
        conv2d_61 = torch.conv2d(
            input_21,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_21 = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_22 = torch.nn.functional.batch_norm(
            conv2d_61,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_61 = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_23 = x_35 + input_22
        x_35 = input_22 = None
        conv2d_62 = torch.conv2d(
            input_23,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_62 = torch.nn.functional.batch_norm(
            conv2d_62,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_62 = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_5 = batch_norm_62.flatten(2)
        batch_norm_62 = None
        qkv_10 = flatten_5.transpose(1, 2)
        flatten_5 = None
        qkv_11 = qkv_10.reshape(4, 400, 768)
        qkv_10 = None
        view_5 = qkv_11.view(4, 400, 8, 96)
        qkv_11 = None
        permute_25 = view_5.permute(0, 2, 3, 1)
        view_5 = None
        split_5 = permute_25.split([32, 32, 32], dim=2)
        permute_25 = None
        q_5 = split_5[0]
        k_5 = split_5[1]
        v_20 = split_5[2]
        split_5 = None
        transpose_16 = q_5.transpose(-2, -1)
        q_5 = None
        matmul_10 = transpose_16 @ k_5
        transpose_16 = k_5 = None
        attn_10 = matmul_10 * 0.1767766952966369
        matmul_10 = None
        attn_11 = attn_10.softmax(dim=-1)
        attn_10 = None
        transpose_17 = attn_11.transpose(-2, -1)
        attn_11 = None
        x_36 = v_20 @ transpose_17
        transpose_17 = None
        x_37 = x_36.permute(0, 3, 1, 2)
        x_36 = None
        v_21 = v_20.permute(0, 3, 1, 2)
        v_20 = None
        x_38 = x_37.reshape(1, 1600, 256)
        x_37 = None
        v_22 = v_21.reshape(1, 1600, 256)
        v_21 = None
        reshape_28 = x_38.reshape(1, 40, 40, 256)
        x_38 = None
        permute_28 = reshape_28.permute(0, 3, 1, 2)
        reshape_28 = None
        x_39 = permute_28.contiguous()
        permute_28 = None
        reshape_29 = v_22.reshape(1, 40, 40, 256)
        v_22 = None
        permute_29 = reshape_29.permute(0, 3, 1, 2)
        reshape_29 = None
        v_23 = permute_29.contiguous()
        permute_29 = None
        conv2d_63 = torch.conv2d(
            v_23,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_23 = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_63 = torch.nn.functional.batch_norm(
            conv2d_63,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_63 = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_40 = x_39 + batch_norm_63
        x_39 = batch_norm_63 = None
        conv2d_64 = torch.conv2d(
            x_40,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_64 = torch.nn.functional.batch_norm(
            conv2d_64,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_64 = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_41 = input_23 + batch_norm_64
        input_23 = batch_norm_64 = None
        conv2d_65 = torch.conv2d(
            x_41,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_65 = torch.nn.functional.batch_norm(
            conv2d_65,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_65 = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_24 = torch.nn.functional.silu(batch_norm_65, inplace=False)
        batch_norm_65 = None
        conv2d_66 = torch.conv2d(
            input_24,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_24 = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_25 = torch.nn.functional.batch_norm(
            conv2d_66,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_66 = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_26 = x_41 + input_25
        x_41 = input_25 = None
        conv2d_67 = torch.conv2d(
            silu_36,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_67 = torch.nn.functional.batch_norm(
            conv2d_67,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_67 = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_6 = batch_norm_67.flatten(2)
        batch_norm_67 = None
        qkv_12 = flatten_6.transpose(1, 2)
        flatten_6 = None
        qkv_13 = qkv_12.reshape(4, 400, 768)
        qkv_12 = None
        view_6 = qkv_13.view(4, 400, 8, 96)
        qkv_13 = None
        permute_30 = view_6.permute(0, 2, 3, 1)
        view_6 = None
        split_6 = permute_30.split([32, 32, 32], dim=2)
        permute_30 = None
        q_6 = split_6[0]
        k_6 = split_6[1]
        v_24 = split_6[2]
        split_6 = None
        transpose_19 = q_6.transpose(-2, -1)
        q_6 = None
        matmul_12 = transpose_19 @ k_6
        transpose_19 = k_6 = None
        attn_12 = matmul_12 * 0.1767766952966369
        matmul_12 = None
        attn_13 = attn_12.softmax(dim=-1)
        attn_12 = None
        transpose_20 = attn_13.transpose(-2, -1)
        attn_13 = None
        x_42 = v_24 @ transpose_20
        transpose_20 = None
        x_43 = x_42.permute(0, 3, 1, 2)
        x_42 = None
        v_25 = v_24.permute(0, 3, 1, 2)
        v_24 = None
        x_44 = x_43.reshape(1, 1600, 256)
        x_43 = None
        v_26 = v_25.reshape(1, 1600, 256)
        v_25 = None
        reshape_33 = x_44.reshape(1, 40, 40, 256)
        x_44 = None
        permute_33 = reshape_33.permute(0, 3, 1, 2)
        reshape_33 = None
        x_45 = permute_33.contiguous()
        permute_33 = None
        reshape_34 = v_26.reshape(1, 40, 40, 256)
        v_26 = None
        permute_34 = reshape_34.permute(0, 3, 1, 2)
        reshape_34 = None
        v_27 = permute_34.contiguous()
        permute_34 = None
        conv2d_68 = torch.conv2d(
            v_27,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_27 = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_68 = torch.nn.functional.batch_norm(
            conv2d_68,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_68 = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_46 = x_45 + batch_norm_68
        x_45 = batch_norm_68 = None
        conv2d_69 = torch.conv2d(
            x_46,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_69 = torch.nn.functional.batch_norm(
            conv2d_69,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_69 = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_47 = silu_36 + batch_norm_69
        batch_norm_69 = None
        conv2d_70 = torch.conv2d(
            x_47,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_70 = torch.nn.functional.batch_norm(
            conv2d_70,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_70 = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_27 = torch.nn.functional.silu(batch_norm_70, inplace=False)
        batch_norm_70 = None
        conv2d_71 = torch.conv2d(
            input_27,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_27 = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_28 = torch.nn.functional.batch_norm(
            conv2d_71,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_71 = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_29 = x_47 + input_28
        x_47 = input_28 = None
        conv2d_72 = torch.conv2d(
            input_29,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_72 = torch.nn.functional.batch_norm(
            conv2d_72,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_72 = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_7 = batch_norm_72.flatten(2)
        batch_norm_72 = None
        qkv_14 = flatten_7.transpose(1, 2)
        flatten_7 = None
        qkv_15 = qkv_14.reshape(4, 400, 768)
        qkv_14 = None
        view_7 = qkv_15.view(4, 400, 8, 96)
        qkv_15 = None
        permute_35 = view_7.permute(0, 2, 3, 1)
        view_7 = None
        split_7 = permute_35.split([32, 32, 32], dim=2)
        permute_35 = None
        q_7 = split_7[0]
        k_7 = split_7[1]
        v_28 = split_7[2]
        split_7 = None
        transpose_22 = q_7.transpose(-2, -1)
        q_7 = None
        matmul_14 = transpose_22 @ k_7
        transpose_22 = k_7 = None
        attn_14 = matmul_14 * 0.1767766952966369
        matmul_14 = None
        attn_15 = attn_14.softmax(dim=-1)
        attn_14 = None
        transpose_23 = attn_15.transpose(-2, -1)
        attn_15 = None
        x_48 = v_28 @ transpose_23
        transpose_23 = None
        x_49 = x_48.permute(0, 3, 1, 2)
        x_48 = None
        v_29 = v_28.permute(0, 3, 1, 2)
        v_28 = None
        x_50 = x_49.reshape(1, 1600, 256)
        x_49 = None
        v_30 = v_29.reshape(1, 1600, 256)
        v_29 = None
        reshape_38 = x_50.reshape(1, 40, 40, 256)
        x_50 = None
        permute_38 = reshape_38.permute(0, 3, 1, 2)
        reshape_38 = None
        x_51 = permute_38.contiguous()
        permute_38 = None
        reshape_39 = v_30.reshape(1, 40, 40, 256)
        v_30 = None
        permute_39 = reshape_39.permute(0, 3, 1, 2)
        reshape_39 = None
        v_31 = permute_39.contiguous()
        permute_39 = None
        conv2d_73 = torch.conv2d(
            v_31,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_31 = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_73 = torch.nn.functional.batch_norm(
            conv2d_73,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_73 = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_52 = x_51 + batch_norm_73
        x_51 = batch_norm_73 = None
        conv2d_74 = torch.conv2d(
            x_52,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_74 = torch.nn.functional.batch_norm(
            conv2d_74,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_74 = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_53 = input_29 + batch_norm_74
        input_29 = batch_norm_74 = None
        conv2d_75 = torch.conv2d(
            x_53,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_75 = torch.nn.functional.batch_norm(
            conv2d_75,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_75 = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_30 = torch.nn.functional.silu(batch_norm_75, inplace=False)
        batch_norm_75 = None
        conv2d_76 = torch.conv2d(
            input_30,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_30 = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_31 = torch.nn.functional.batch_norm(
            conv2d_76,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_76 = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_32 = x_53 + input_31
        x_53 = input_31 = None
        cat_6 = torch.cat([silu_36, input_14, input_20, input_26, input_32], 1)
        silu_36 = input_14 = input_20 = input_26 = input_32 = None
        conv2d_77 = torch.conv2d(
            cat_6,
            l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_6 = (
            l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_77 = torch.nn.functional.batch_norm(
            conv2d_77,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_77 = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_
        ) = None
        silu_45 = torch.nn.functional.silu(batch_norm_77, inplace=False)
        batch_norm_77 = None
        view_8 = l_self_modules_model_modules_6_parameters_gamma_.view(-1, 512, 1, 1)
        l_self_modules_model_modules_6_parameters_gamma_ = None
        mul_8 = view_8 * silu_45
        view_8 = silu_45 = None
        x_54 = x_5 + mul_8
        x_5 = mul_8 = None
        conv2d_78 = torch.conv2d(
            x_54,
            l_self_modules_model_modules_7_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_model_modules_7_modules_conv_parameters_weight_ = None
        batch_norm_78 = torch.nn.functional.batch_norm(
            conv2d_78,
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_7_modules_bn_parameters_weight_,
            l_self_modules_model_modules_7_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_78 = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_7_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_7_modules_bn_parameters_bias_ = None
        x_55 = torch.nn.functional.silu(batch_norm_78, inplace=False)
        batch_norm_78 = None
        conv2d_79 = torch.conv2d(
            x_55,
            l_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_79 = torch.nn.functional.batch_norm(
            conv2d_79,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_79 = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_47 = torch.nn.functional.silu(batch_norm_79, inplace=False)
        batch_norm_79 = None
        conv2d_80 = torch.conv2d(
            silu_47,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_80 = torch.nn.functional.batch_norm(
            conv2d_80,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_80 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_8 = batch_norm_80.flatten(2)
        batch_norm_80 = None
        qkv_16 = flatten_8.transpose(1, 2)
        flatten_8 = None
        view_9 = qkv_16.view(1, 400, 8, 96)
        qkv_16 = None
        permute_40 = view_9.permute(0, 2, 3, 1)
        view_9 = None
        split_8 = permute_40.split([32, 32, 32], dim=2)
        permute_40 = None
        q_8 = split_8[0]
        k_8 = split_8[1]
        v_32 = split_8[2]
        split_8 = None
        transpose_25 = q_8.transpose(-2, -1)
        q_8 = None
        matmul_16 = transpose_25 @ k_8
        transpose_25 = k_8 = None
        attn_16 = matmul_16 * 0.1767766952966369
        matmul_16 = None
        attn_17 = attn_16.softmax(dim=-1)
        attn_16 = None
        transpose_26 = attn_17.transpose(-2, -1)
        attn_17 = None
        x_56 = v_32 @ transpose_26
        transpose_26 = None
        x_57 = x_56.permute(0, 3, 1, 2)
        x_56 = None
        v_33 = v_32.permute(0, 3, 1, 2)
        v_32 = None
        reshape_40 = x_57.reshape(1, 20, 20, 256)
        x_57 = None
        permute_43 = reshape_40.permute(0, 3, 1, 2)
        reshape_40 = None
        x_58 = permute_43.contiguous()
        permute_43 = None
        reshape_41 = v_33.reshape(1, 20, 20, 256)
        v_33 = None
        permute_44 = reshape_41.permute(0, 3, 1, 2)
        reshape_41 = None
        v_34 = permute_44.contiguous()
        permute_44 = None
        conv2d_81 = torch.conv2d(
            v_34,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_34 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_81 = torch.nn.functional.batch_norm(
            conv2d_81,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_81 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_59 = x_58 + batch_norm_81
        x_58 = batch_norm_81 = None
        conv2d_82 = torch.conv2d(
            x_59,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_82 = torch.nn.functional.batch_norm(
            conv2d_82,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_82 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_60 = silu_47 + batch_norm_82
        batch_norm_82 = None
        conv2d_83 = torch.conv2d(
            x_60,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_83 = torch.nn.functional.batch_norm(
            conv2d_83,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_83 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_33 = torch.nn.functional.silu(batch_norm_83, inplace=False)
        batch_norm_83 = None
        conv2d_84 = torch.conv2d(
            input_33,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_33 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_34 = torch.nn.functional.batch_norm(
            conv2d_84,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_84 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_35 = x_60 + input_34
        x_60 = input_34 = None
        conv2d_85 = torch.conv2d(
            input_35,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_85 = torch.nn.functional.batch_norm(
            conv2d_85,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_85 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_9 = batch_norm_85.flatten(2)
        batch_norm_85 = None
        qkv_17 = flatten_9.transpose(1, 2)
        flatten_9 = None
        view_10 = qkv_17.view(1, 400, 8, 96)
        qkv_17 = None
        permute_45 = view_10.permute(0, 2, 3, 1)
        view_10 = None
        split_9 = permute_45.split([32, 32, 32], dim=2)
        permute_45 = None
        q_9 = split_9[0]
        k_9 = split_9[1]
        v_35 = split_9[2]
        split_9 = None
        transpose_28 = q_9.transpose(-2, -1)
        q_9 = None
        matmul_18 = transpose_28 @ k_9
        transpose_28 = k_9 = None
        attn_18 = matmul_18 * 0.1767766952966369
        matmul_18 = None
        attn_19 = attn_18.softmax(dim=-1)
        attn_18 = None
        transpose_29 = attn_19.transpose(-2, -1)
        attn_19 = None
        x_61 = v_35 @ transpose_29
        transpose_29 = None
        x_62 = x_61.permute(0, 3, 1, 2)
        x_61 = None
        v_36 = v_35.permute(0, 3, 1, 2)
        v_35 = None
        reshape_42 = x_62.reshape(1, 20, 20, 256)
        x_62 = None
        permute_48 = reshape_42.permute(0, 3, 1, 2)
        reshape_42 = None
        x_63 = permute_48.contiguous()
        permute_48 = None
        reshape_43 = v_36.reshape(1, 20, 20, 256)
        v_36 = None
        permute_49 = reshape_43.permute(0, 3, 1, 2)
        reshape_43 = None
        v_37 = permute_49.contiguous()
        permute_49 = None
        conv2d_86 = torch.conv2d(
            v_37,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_37 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_86 = torch.nn.functional.batch_norm(
            conv2d_86,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_86 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_64 = x_63 + batch_norm_86
        x_63 = batch_norm_86 = None
        conv2d_87 = torch.conv2d(
            x_64,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_87 = torch.nn.functional.batch_norm(
            conv2d_87,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_87 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_65 = input_35 + batch_norm_87
        input_35 = batch_norm_87 = None
        conv2d_88 = torch.conv2d(
            x_65,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_88 = torch.nn.functional.batch_norm(
            conv2d_88,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_88 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_36 = torch.nn.functional.silu(batch_norm_88, inplace=False)
        batch_norm_88 = None
        conv2d_89 = torch.conv2d(
            input_36,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_36 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_37 = torch.nn.functional.batch_norm(
            conv2d_89,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_89 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_38 = x_65 + input_37
        x_65 = input_37 = None
        conv2d_90 = torch.conv2d(
            silu_47,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_90 = torch.nn.functional.batch_norm(
            conv2d_90,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_90 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_10 = batch_norm_90.flatten(2)
        batch_norm_90 = None
        qkv_18 = flatten_10.transpose(1, 2)
        flatten_10 = None
        view_11 = qkv_18.view(1, 400, 8, 96)
        qkv_18 = None
        permute_50 = view_11.permute(0, 2, 3, 1)
        view_11 = None
        split_10 = permute_50.split([32, 32, 32], dim=2)
        permute_50 = None
        q_10 = split_10[0]
        k_10 = split_10[1]
        v_38 = split_10[2]
        split_10 = None
        transpose_31 = q_10.transpose(-2, -1)
        q_10 = None
        matmul_20 = transpose_31 @ k_10
        transpose_31 = k_10 = None
        attn_20 = matmul_20 * 0.1767766952966369
        matmul_20 = None
        attn_21 = attn_20.softmax(dim=-1)
        attn_20 = None
        transpose_32 = attn_21.transpose(-2, -1)
        attn_21 = None
        x_66 = v_38 @ transpose_32
        transpose_32 = None
        x_67 = x_66.permute(0, 3, 1, 2)
        x_66 = None
        v_39 = v_38.permute(0, 3, 1, 2)
        v_38 = None
        reshape_44 = x_67.reshape(1, 20, 20, 256)
        x_67 = None
        permute_53 = reshape_44.permute(0, 3, 1, 2)
        reshape_44 = None
        x_68 = permute_53.contiguous()
        permute_53 = None
        reshape_45 = v_39.reshape(1, 20, 20, 256)
        v_39 = None
        permute_54 = reshape_45.permute(0, 3, 1, 2)
        reshape_45 = None
        v_40 = permute_54.contiguous()
        permute_54 = None
        conv2d_91 = torch.conv2d(
            v_40,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_40 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_91 = torch.nn.functional.batch_norm(
            conv2d_91,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_91 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_69 = x_68 + batch_norm_91
        x_68 = batch_norm_91 = None
        conv2d_92 = torch.conv2d(
            x_69,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_92 = torch.nn.functional.batch_norm(
            conv2d_92,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_92 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_70 = silu_47 + batch_norm_92
        batch_norm_92 = None
        conv2d_93 = torch.conv2d(
            x_70,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_93 = torch.nn.functional.batch_norm(
            conv2d_93,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_93 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_39 = torch.nn.functional.silu(batch_norm_93, inplace=False)
        batch_norm_93 = None
        conv2d_94 = torch.conv2d(
            input_39,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_39 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_40 = torch.nn.functional.batch_norm(
            conv2d_94,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_94 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_41 = x_70 + input_40
        x_70 = input_40 = None
        conv2d_95 = torch.conv2d(
            input_41,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_95 = torch.nn.functional.batch_norm(
            conv2d_95,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_95 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_11 = batch_norm_95.flatten(2)
        batch_norm_95 = None
        qkv_19 = flatten_11.transpose(1, 2)
        flatten_11 = None
        view_12 = qkv_19.view(1, 400, 8, 96)
        qkv_19 = None
        permute_55 = view_12.permute(0, 2, 3, 1)
        view_12 = None
        split_11 = permute_55.split([32, 32, 32], dim=2)
        permute_55 = None
        q_11 = split_11[0]
        k_11 = split_11[1]
        v_41 = split_11[2]
        split_11 = None
        transpose_34 = q_11.transpose(-2, -1)
        q_11 = None
        matmul_22 = transpose_34 @ k_11
        transpose_34 = k_11 = None
        attn_22 = matmul_22 * 0.1767766952966369
        matmul_22 = None
        attn_23 = attn_22.softmax(dim=-1)
        attn_22 = None
        transpose_35 = attn_23.transpose(-2, -1)
        attn_23 = None
        x_71 = v_41 @ transpose_35
        transpose_35 = None
        x_72 = x_71.permute(0, 3, 1, 2)
        x_71 = None
        v_42 = v_41.permute(0, 3, 1, 2)
        v_41 = None
        reshape_46 = x_72.reshape(1, 20, 20, 256)
        x_72 = None
        permute_58 = reshape_46.permute(0, 3, 1, 2)
        reshape_46 = None
        x_73 = permute_58.contiguous()
        permute_58 = None
        reshape_47 = v_42.reshape(1, 20, 20, 256)
        v_42 = None
        permute_59 = reshape_47.permute(0, 3, 1, 2)
        reshape_47 = None
        v_43 = permute_59.contiguous()
        permute_59 = None
        conv2d_96 = torch.conv2d(
            v_43,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_43 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_96 = torch.nn.functional.batch_norm(
            conv2d_96,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_96 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_74 = x_73 + batch_norm_96
        x_73 = batch_norm_96 = None
        conv2d_97 = torch.conv2d(
            x_74,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_97 = torch.nn.functional.batch_norm(
            conv2d_97,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_97 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_75 = input_41 + batch_norm_97
        input_41 = batch_norm_97 = None
        conv2d_98 = torch.conv2d(
            x_75,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_98 = torch.nn.functional.batch_norm(
            conv2d_98,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_98 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_42 = torch.nn.functional.silu(batch_norm_98, inplace=False)
        batch_norm_98 = None
        conv2d_99 = torch.conv2d(
            input_42,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_42 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_43 = torch.nn.functional.batch_norm(
            conv2d_99,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_99 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_44 = x_75 + input_43
        x_75 = input_43 = None
        conv2d_100 = torch.conv2d(
            silu_47,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_100 = torch.nn.functional.batch_norm(
            conv2d_100,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_100 = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_12 = batch_norm_100.flatten(2)
        batch_norm_100 = None
        qkv_20 = flatten_12.transpose(1, 2)
        flatten_12 = None
        view_13 = qkv_20.view(1, 400, 8, 96)
        qkv_20 = None
        permute_60 = view_13.permute(0, 2, 3, 1)
        view_13 = None
        split_12 = permute_60.split([32, 32, 32], dim=2)
        permute_60 = None
        q_12 = split_12[0]
        k_12 = split_12[1]
        v_44 = split_12[2]
        split_12 = None
        transpose_37 = q_12.transpose(-2, -1)
        q_12 = None
        matmul_24 = transpose_37 @ k_12
        transpose_37 = k_12 = None
        attn_24 = matmul_24 * 0.1767766952966369
        matmul_24 = None
        attn_25 = attn_24.softmax(dim=-1)
        attn_24 = None
        transpose_38 = attn_25.transpose(-2, -1)
        attn_25 = None
        x_76 = v_44 @ transpose_38
        transpose_38 = None
        x_77 = x_76.permute(0, 3, 1, 2)
        x_76 = None
        v_45 = v_44.permute(0, 3, 1, 2)
        v_44 = None
        reshape_48 = x_77.reshape(1, 20, 20, 256)
        x_77 = None
        permute_63 = reshape_48.permute(0, 3, 1, 2)
        reshape_48 = None
        x_78 = permute_63.contiguous()
        permute_63 = None
        reshape_49 = v_45.reshape(1, 20, 20, 256)
        v_45 = None
        permute_64 = reshape_49.permute(0, 3, 1, 2)
        reshape_49 = None
        v_46 = permute_64.contiguous()
        permute_64 = None
        conv2d_101 = torch.conv2d(
            v_46,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_46 = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_101 = torch.nn.functional.batch_norm(
            conv2d_101,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_101 = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_79 = x_78 + batch_norm_101
        x_78 = batch_norm_101 = None
        conv2d_102 = torch.conv2d(
            x_79,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_102 = torch.nn.functional.batch_norm(
            conv2d_102,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_102 = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_80 = silu_47 + batch_norm_102
        batch_norm_102 = None
        conv2d_103 = torch.conv2d(
            x_80,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_103 = torch.nn.functional.batch_norm(
            conv2d_103,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_103 = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_45 = torch.nn.functional.silu(batch_norm_103, inplace=False)
        batch_norm_103 = None
        conv2d_104 = torch.conv2d(
            input_45,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_45 = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_46 = torch.nn.functional.batch_norm(
            conv2d_104,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_104 = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_47 = x_80 + input_46
        x_80 = input_46 = None
        conv2d_105 = torch.conv2d(
            input_47,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_105 = torch.nn.functional.batch_norm(
            conv2d_105,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_105 = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_13 = batch_norm_105.flatten(2)
        batch_norm_105 = None
        qkv_21 = flatten_13.transpose(1, 2)
        flatten_13 = None
        view_14 = qkv_21.view(1, 400, 8, 96)
        qkv_21 = None
        permute_65 = view_14.permute(0, 2, 3, 1)
        view_14 = None
        split_13 = permute_65.split([32, 32, 32], dim=2)
        permute_65 = None
        q_13 = split_13[0]
        k_13 = split_13[1]
        v_47 = split_13[2]
        split_13 = None
        transpose_40 = q_13.transpose(-2, -1)
        q_13 = None
        matmul_26 = transpose_40 @ k_13
        transpose_40 = k_13 = None
        attn_26 = matmul_26 * 0.1767766952966369
        matmul_26 = None
        attn_27 = attn_26.softmax(dim=-1)
        attn_26 = None
        transpose_41 = attn_27.transpose(-2, -1)
        attn_27 = None
        x_81 = v_47 @ transpose_41
        transpose_41 = None
        x_82 = x_81.permute(0, 3, 1, 2)
        x_81 = None
        v_48 = v_47.permute(0, 3, 1, 2)
        v_47 = None
        reshape_50 = x_82.reshape(1, 20, 20, 256)
        x_82 = None
        permute_68 = reshape_50.permute(0, 3, 1, 2)
        reshape_50 = None
        x_83 = permute_68.contiguous()
        permute_68 = None
        reshape_51 = v_48.reshape(1, 20, 20, 256)
        v_48 = None
        permute_69 = reshape_51.permute(0, 3, 1, 2)
        reshape_51 = None
        v_49 = permute_69.contiguous()
        permute_69 = None
        conv2d_106 = torch.conv2d(
            v_49,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_49 = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_106 = torch.nn.functional.batch_norm(
            conv2d_106,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_106 = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_84 = x_83 + batch_norm_106
        x_83 = batch_norm_106 = None
        conv2d_107 = torch.conv2d(
            x_84,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_107 = torch.nn.functional.batch_norm(
            conv2d_107,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_107 = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_85 = input_47 + batch_norm_107
        input_47 = batch_norm_107 = None
        conv2d_108 = torch.conv2d(
            x_85,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_108 = torch.nn.functional.batch_norm(
            conv2d_108,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_108 = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_48 = torch.nn.functional.silu(batch_norm_108, inplace=False)
        batch_norm_108 = None
        conv2d_109 = torch.conv2d(
            input_48,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_48 = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_49 = torch.nn.functional.batch_norm(
            conv2d_109,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_109 = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_2_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_50 = x_85 + input_49
        x_85 = input_49 = None
        conv2d_110 = torch.conv2d(
            silu_47,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_110 = torch.nn.functional.batch_norm(
            conv2d_110,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_110 = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_14 = batch_norm_110.flatten(2)
        batch_norm_110 = None
        qkv_22 = flatten_14.transpose(1, 2)
        flatten_14 = None
        view_15 = qkv_22.view(1, 400, 8, 96)
        qkv_22 = None
        permute_70 = view_15.permute(0, 2, 3, 1)
        view_15 = None
        split_14 = permute_70.split([32, 32, 32], dim=2)
        permute_70 = None
        q_14 = split_14[0]
        k_14 = split_14[1]
        v_50 = split_14[2]
        split_14 = None
        transpose_43 = q_14.transpose(-2, -1)
        q_14 = None
        matmul_28 = transpose_43 @ k_14
        transpose_43 = k_14 = None
        attn_28 = matmul_28 * 0.1767766952966369
        matmul_28 = None
        attn_29 = attn_28.softmax(dim=-1)
        attn_28 = None
        transpose_44 = attn_29.transpose(-2, -1)
        attn_29 = None
        x_86 = v_50 @ transpose_44
        transpose_44 = None
        x_87 = x_86.permute(0, 3, 1, 2)
        x_86 = None
        v_51 = v_50.permute(0, 3, 1, 2)
        v_50 = None
        reshape_52 = x_87.reshape(1, 20, 20, 256)
        x_87 = None
        permute_73 = reshape_52.permute(0, 3, 1, 2)
        reshape_52 = None
        x_88 = permute_73.contiguous()
        permute_73 = None
        reshape_53 = v_51.reshape(1, 20, 20, 256)
        v_51 = None
        permute_74 = reshape_53.permute(0, 3, 1, 2)
        reshape_53 = None
        v_52 = permute_74.contiguous()
        permute_74 = None
        conv2d_111 = torch.conv2d(
            v_52,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_52 = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_111 = torch.nn.functional.batch_norm(
            conv2d_111,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_111 = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_89 = x_88 + batch_norm_111
        x_88 = batch_norm_111 = None
        conv2d_112 = torch.conv2d(
            x_89,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_112 = torch.nn.functional.batch_norm(
            conv2d_112,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_112 = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_90 = silu_47 + batch_norm_112
        batch_norm_112 = None
        conv2d_113 = torch.conv2d(
            x_90,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_113 = torch.nn.functional.batch_norm(
            conv2d_113,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_113 = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_51 = torch.nn.functional.silu(batch_norm_113, inplace=False)
        batch_norm_113 = None
        conv2d_114 = torch.conv2d(
            input_51,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_51 = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_52 = torch.nn.functional.batch_norm(
            conv2d_114,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_114 = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_53 = x_90 + input_52
        x_90 = input_52 = None
        conv2d_115 = torch.conv2d(
            input_53,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_115 = torch.nn.functional.batch_norm(
            conv2d_115,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_115 = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        flatten_15 = batch_norm_115.flatten(2)
        batch_norm_115 = None
        qkv_23 = flatten_15.transpose(1, 2)
        flatten_15 = None
        view_16 = qkv_23.view(1, 400, 8, 96)
        qkv_23 = None
        permute_75 = view_16.permute(0, 2, 3, 1)
        view_16 = None
        split_15 = permute_75.split([32, 32, 32], dim=2)
        permute_75 = None
        q_15 = split_15[0]
        k_15 = split_15[1]
        v_53 = split_15[2]
        split_15 = None
        transpose_46 = q_15.transpose(-2, -1)
        q_15 = None
        matmul_30 = transpose_46 @ k_15
        transpose_46 = k_15 = None
        attn_30 = matmul_30 * 0.1767766952966369
        matmul_30 = None
        attn_31 = attn_30.softmax(dim=-1)
        attn_30 = None
        transpose_47 = attn_31.transpose(-2, -1)
        attn_31 = None
        x_91 = v_53 @ transpose_47
        transpose_47 = None
        x_92 = x_91.permute(0, 3, 1, 2)
        x_91 = None
        v_54 = v_53.permute(0, 3, 1, 2)
        v_53 = None
        reshape_54 = x_92.reshape(1, 20, 20, 256)
        x_92 = None
        permute_78 = reshape_54.permute(0, 3, 1, 2)
        reshape_54 = None
        x_93 = permute_78.contiguous()
        permute_78 = None
        reshape_55 = v_54.reshape(1, 20, 20, 256)
        v_54 = None
        permute_79 = reshape_55.permute(0, 3, 1, 2)
        reshape_55 = None
        v_55 = permute_79.contiguous()
        permute_79 = None
        conv2d_116 = torch.conv2d(
            v_55,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        v_55 = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_116 = torch.nn.functional.batch_norm(
            conv2d_116,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_116 = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_94 = x_93 + batch_norm_116
        x_93 = batch_norm_116 = None
        conv2d_117 = torch.conv2d(
            x_94,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        batch_norm_117 = torch.nn.functional.batch_norm(
            conv2d_117,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_117 = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_95 = input_53 + batch_norm_117
        input_53 = batch_norm_117 = None
        conv2d_118 = torch.conv2d(
            x_95,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_118 = torch.nn.functional.batch_norm(
            conv2d_118,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_118 = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (None)
        input_54 = torch.nn.functional.silu(batch_norm_118, inplace=False)
        batch_norm_118 = None
        conv2d_119 = torch.conv2d(
            input_54,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_54 = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        input_55 = torch.nn.functional.batch_norm(
            conv2d_119,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_119 = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_3_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (None)
        input_56 = x_95 + input_55
        x_95 = input_55 = None
        cat_7 = torch.cat([silu_47, input_38, input_44, input_50, input_56], 1)
        silu_47 = input_38 = input_44 = input_50 = input_56 = None
        conv2d_120 = torch.conv2d(
            cat_7,
            l_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_7 = (
            l_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_120 = torch.nn.functional.batch_norm(
            conv2d_120,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_120 = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_
        ) = None
        silu_56 = torch.nn.functional.silu(batch_norm_120, inplace=False)
        batch_norm_120 = None
        view_17 = l_self_modules_model_modules_8_parameters_gamma_.view(-1, 512, 1, 1)
        l_self_modules_model_modules_8_parameters_gamma_ = None
        mul_17 = view_17 * silu_56
        view_17 = silu_56 = None
        x_96 = x_55 + mul_17
        x_55 = mul_17 = None
        conv2d_121 = torch.conv2d(
            x_96,
            l_self_modules_model_modules_9_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = (
            l_self_modules_model_modules_9_modules_conv_modules_conv_parameters_weight_
        ) = None
        batch_norm_121 = torch.nn.functional.batch_norm(
            conv2d_121,
            l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_121 = (
            l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_bias_
        ) = None
        silu_57 = torch.nn.functional.silu(batch_norm_121, inplace=False)
        batch_norm_121 = None
        adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d(silu_57, 1)
        silu_57 = None
        flatten_16 = adaptive_avg_pool2d.flatten(1)
        adaptive_avg_pool2d = None
        dropout = torch.nn.functional.dropout(flatten_16, 0.0, False, True)
        flatten_16 = None
        x_97 = torch._C._nn.linear(
            dropout,
            l_self_modules_model_modules_9_modules_linear_parameters_weight_,
            l_self_modules_model_modules_9_modules_linear_parameters_bias_,
        )
        dropout = (
            l_self_modules_model_modules_9_modules_linear_parameters_weight_
        ) = l_self_modules_model_modules_9_modules_linear_parameters_bias_ = None
        y = x_97.softmax(1)
        return (y, x_97)
