import torch

from torch import device


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
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_17_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_17_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_17_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_17_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_17_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_20_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_20_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_stride: torch.Tensor,
        L_self_modules_model_modules_23_modules_dfl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_
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
        l_self_modules_model_modules_10_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_10_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_10_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_10_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_13_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_13_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_13_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_16_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_16_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_16_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_17_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_17_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_17_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_17_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_17_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_17_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_17_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_17_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_17_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_17_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_20_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_20_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_20_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_20_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_20_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_20_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_20_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_20_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_20_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_20_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_22_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_22_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_22_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_bias_ = L_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_bias_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_bias_ = L_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_bias_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_weight_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_weight_
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_bias_ = L_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_weight_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_weight_
        l_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_bias_ = L_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_bias_
        l_self_modules_model_modules_23_stride = L_self_modules_model_modules_23_stride
        l_self_modules_model_modules_23_modules_dfl_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_dfl_modules_conv_parameters_weight_
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
            0.03,
            0.001,
        )
        conv2d = (
            l_self_modules_model_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_0_modules_bn_parameters_bias_ = None
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
        batch_norm_1 = torch.nn.functional.batch_norm(
            conv2d_1,
            l_self_modules_model_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_1 = (
            l_self_modules_model_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_1_modules_bn_parameters_bias_ = None
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
        batch_norm_2 = torch.nn.functional.batch_norm(
            conv2d_2,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
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
            0.03,
            0.001,
        )
        conv2d_3 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_3 = torch.nn.functional.silu(batch_norm_3, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_4 = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_4 = torch.nn.functional.silu(batch_norm_4, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_5 = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_5 = torch.nn.functional.silu(batch_norm_5, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_6 = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_6 = torch.nn.functional.silu(batch_norm_6, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_7 = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_7 = torch.nn.functional.silu(batch_norm_7, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_8 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_8 = torch.nn.functional.silu(batch_norm_8, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_9 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_9 = torch.nn.functional.silu(batch_norm_9, inplace=True)
        batch_norm_9 = None
        cat_1 = torch.cat([getitem, getitem_1, silu_9], 1)
        getitem = getitem_1 = silu_9 = None
        conv2d_10 = torch.conv2d(
            cat_1,
            l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = (
            l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_10 = torch.nn.functional.batch_norm(
            conv2d_10,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_10 = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_2 = torch.nn.functional.silu(batch_norm_10, inplace=True)
        batch_norm_10 = None
        conv2d_11 = torch.conv2d(
            x_2,
            l_self_modules_model_modules_3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_model_modules_3_modules_conv_parameters_weight_ = None
        batch_norm_11 = torch.nn.functional.batch_norm(
            conv2d_11,
            l_self_modules_model_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_11 = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_3_modules_bn_parameters_bias_ = None
        x_3 = torch.nn.functional.silu(batch_norm_11, inplace=True)
        batch_norm_11 = None
        conv2d_12 = torch.conv2d(
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
        batch_norm_12 = torch.nn.functional.batch_norm(
            conv2d_12,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_12 = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_12 = torch.nn.functional.silu(batch_norm_12, inplace=True)
        batch_norm_12 = None
        chunk_1 = silu_12.chunk(2, 1)
        silu_12 = None
        getitem_2 = chunk_1[0]
        getitem_3 = chunk_1[1]
        chunk_1 = None
        conv2d_13 = torch.conv2d(
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
        batch_norm_13 = torch.nn.functional.batch_norm(
            conv2d_13,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_13 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_13 = torch.nn.functional.silu(batch_norm_13, inplace=True)
        batch_norm_13 = None
        conv2d_14 = torch.conv2d(
            silu_13,
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
        batch_norm_14 = torch.nn.functional.batch_norm(
            conv2d_14,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_14 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_14 = torch.nn.functional.silu(batch_norm_14, inplace=True)
        batch_norm_14 = None
        conv2d_15 = torch.conv2d(
            silu_14,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_14 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_15 = torch.nn.functional.batch_norm(
            conv2d_15,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_15 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_15 = torch.nn.functional.silu(batch_norm_15, inplace=True)
        batch_norm_15 = None
        input_3 = silu_13 + silu_15
        silu_13 = silu_15 = None
        conv2d_16 = torch.conv2d(
            input_3,
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
        batch_norm_16 = torch.nn.functional.batch_norm(
            conv2d_16,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_16 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_16 = torch.nn.functional.silu(batch_norm_16, inplace=True)
        batch_norm_16 = None
        conv2d_17 = torch.conv2d(
            silu_16,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_16 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_17 = torch.nn.functional.batch_norm(
            conv2d_17,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_17 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_17 = torch.nn.functional.silu(batch_norm_17, inplace=True)
        batch_norm_17 = None
        input_4 = input_3 + silu_17
        input_3 = silu_17 = None
        conv2d_18 = torch.conv2d(
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
        batch_norm_18 = torch.nn.functional.batch_norm(
            conv2d_18,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_18 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_18 = torch.nn.functional.silu(batch_norm_18, inplace=True)
        batch_norm_18 = None
        cat_2 = torch.cat((input_4, silu_18), 1)
        input_4 = silu_18 = None
        conv2d_19 = torch.conv2d(
            cat_2,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_19 = torch.nn.functional.batch_norm(
            conv2d_19,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_19 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_19 = torch.nn.functional.silu(batch_norm_19, inplace=True)
        batch_norm_19 = None
        cat_3 = torch.cat([getitem_2, getitem_3, silu_19], 1)
        getitem_2 = getitem_3 = silu_19 = None
        conv2d_20 = torch.conv2d(
            cat_3,
            l_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = (
            l_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_20 = torch.nn.functional.batch_norm(
            conv2d_20,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_20 = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_4 = torch.nn.functional.silu(batch_norm_20, inplace=True)
        batch_norm_20 = None
        conv2d_21 = torch.conv2d(
            x_4,
            l_self_modules_model_modules_5_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_5_modules_conv_parameters_weight_ = None
        batch_norm_21 = torch.nn.functional.batch_norm(
            conv2d_21,
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_5_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_21 = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_5_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_5_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.silu(batch_norm_21, inplace=True)
        batch_norm_21 = None
        conv2d_22 = torch.conv2d(
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
        batch_norm_22 = torch.nn.functional.batch_norm(
            conv2d_22,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_22 = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_22 = torch.nn.functional.silu(batch_norm_22, inplace=True)
        batch_norm_22 = None
        chunk_2 = silu_22.chunk(2, 1)
        silu_22 = None
        getitem_4 = chunk_2[0]
        getitem_5 = chunk_2[1]
        chunk_2 = None
        conv2d_23 = torch.conv2d(
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
        batch_norm_23 = torch.nn.functional.batch_norm(
            conv2d_23,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_23 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_23 = torch.nn.functional.silu(batch_norm_23, inplace=True)
        batch_norm_23 = None
        conv2d_24 = torch.conv2d(
            silu_23,
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
        batch_norm_24 = torch.nn.functional.batch_norm(
            conv2d_24,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_24 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_24 = torch.nn.functional.silu(batch_norm_24, inplace=True)
        batch_norm_24 = None
        conv2d_25 = torch.conv2d(
            silu_24,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_24 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_25 = torch.nn.functional.batch_norm(
            conv2d_25,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_25 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_25 = torch.nn.functional.silu(batch_norm_25, inplace=True)
        batch_norm_25 = None
        input_5 = silu_23 + silu_25
        silu_23 = silu_25 = None
        conv2d_26 = torch.conv2d(
            input_5,
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
        batch_norm_26 = torch.nn.functional.batch_norm(
            conv2d_26,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_26 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_26 = torch.nn.functional.silu(batch_norm_26, inplace=True)
        batch_norm_26 = None
        conv2d_27 = torch.conv2d(
            silu_26,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_26 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_27 = torch.nn.functional.batch_norm(
            conv2d_27,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_27 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_27 = torch.nn.functional.silu(batch_norm_27, inplace=True)
        batch_norm_27 = None
        input_6 = input_5 + silu_27
        input_5 = silu_27 = None
        conv2d_28 = torch.conv2d(
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
        batch_norm_28 = torch.nn.functional.batch_norm(
            conv2d_28,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_28 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_28 = torch.nn.functional.silu(batch_norm_28, inplace=True)
        batch_norm_28 = None
        cat_4 = torch.cat((input_6, silu_28), 1)
        input_6 = silu_28 = None
        conv2d_29 = torch.conv2d(
            cat_4,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_4 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_29 = torch.nn.functional.batch_norm(
            conv2d_29,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_29 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_29 = torch.nn.functional.silu(batch_norm_29, inplace=True)
        batch_norm_29 = None
        cat_5 = torch.cat([getitem_4, getitem_5, silu_29], 1)
        getitem_4 = getitem_5 = silu_29 = None
        conv2d_30 = torch.conv2d(
            cat_5,
            l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_5 = (
            l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_30 = torch.nn.functional.batch_norm(
            conv2d_30,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_30 = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_6 = torch.nn.functional.silu(batch_norm_30, inplace=True)
        batch_norm_30 = None
        conv2d_31 = torch.conv2d(
            x_6,
            l_self_modules_model_modules_7_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_7_modules_conv_parameters_weight_ = None
        batch_norm_31 = torch.nn.functional.batch_norm(
            conv2d_31,
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_7_modules_bn_parameters_weight_,
            l_self_modules_model_modules_7_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_31 = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_7_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_7_modules_bn_parameters_bias_ = None
        x_7 = torch.nn.functional.silu(batch_norm_31, inplace=True)
        batch_norm_31 = None
        conv2d_32 = torch.conv2d(
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
        batch_norm_32 = torch.nn.functional.batch_norm(
            conv2d_32,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_32 = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_32 = torch.nn.functional.silu(batch_norm_32, inplace=True)
        batch_norm_32 = None
        chunk_3 = silu_32.chunk(2, 1)
        silu_32 = None
        getitem_6 = chunk_3[0]
        getitem_7 = chunk_3[1]
        chunk_3 = None
        conv2d_33 = torch.conv2d(
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
        batch_norm_33 = torch.nn.functional.batch_norm(
            conv2d_33,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_33 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_33 = torch.nn.functional.silu(batch_norm_33, inplace=True)
        batch_norm_33 = None
        conv2d_34 = torch.conv2d(
            silu_33,
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
        batch_norm_34 = torch.nn.functional.batch_norm(
            conv2d_34,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_34 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_34 = torch.nn.functional.silu(batch_norm_34, inplace=True)
        batch_norm_34 = None
        conv2d_35 = torch.conv2d(
            silu_34,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_34 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_35 = torch.nn.functional.batch_norm(
            conv2d_35,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_35 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_35 = torch.nn.functional.silu(batch_norm_35, inplace=True)
        batch_norm_35 = None
        input_7 = silu_33 + silu_35
        silu_33 = silu_35 = None
        conv2d_36 = torch.conv2d(
            input_7,
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
        batch_norm_36 = torch.nn.functional.batch_norm(
            conv2d_36,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_36 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_36 = torch.nn.functional.silu(batch_norm_36, inplace=True)
        batch_norm_36 = None
        conv2d_37 = torch.conv2d(
            silu_36,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_36 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_37 = torch.nn.functional.batch_norm(
            conv2d_37,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_37 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_37 = torch.nn.functional.silu(batch_norm_37, inplace=True)
        batch_norm_37 = None
        input_8 = input_7 + silu_37
        input_7 = silu_37 = None
        conv2d_38 = torch.conv2d(
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
        batch_norm_38 = torch.nn.functional.batch_norm(
            conv2d_38,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_38 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_38 = torch.nn.functional.silu(batch_norm_38, inplace=True)
        batch_norm_38 = None
        cat_6 = torch.cat((input_8, silu_38), 1)
        input_8 = silu_38 = None
        conv2d_39 = torch.conv2d(
            cat_6,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_6 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_39 = torch.nn.functional.batch_norm(
            conv2d_39,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_39 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_39 = torch.nn.functional.silu(batch_norm_39, inplace=True)
        batch_norm_39 = None
        cat_7 = torch.cat([getitem_6, getitem_7, silu_39], 1)
        getitem_6 = getitem_7 = silu_39 = None
        conv2d_40 = torch.conv2d(
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
        batch_norm_40 = torch.nn.functional.batch_norm(
            conv2d_40,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_40 = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.silu(batch_norm_40, inplace=True)
        batch_norm_40 = None
        conv2d_41 = torch.conv2d(
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
        batch_norm_41 = torch.nn.functional.batch_norm(
            conv2d_41,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_41 = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_41 = torch.nn.functional.silu(batch_norm_41, inplace=True)
        batch_norm_41 = None
        max_pool2d = torch.nn.functional.max_pool2d(
            silu_41, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_1 = torch.nn.functional.max_pool2d(
            silu_41, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_2 = torch.nn.functional.max_pool2d(
            silu_41, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        cat_8 = torch.cat([silu_41, max_pool2d, max_pool2d_1, max_pool2d_2], 1)
        silu_41 = max_pool2d = max_pool2d_1 = max_pool2d_2 = None
        conv2d_42 = torch.conv2d(
            cat_8,
            l_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_8 = (
            l_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_42 = torch.nn.functional.batch_norm(
            conv2d_42,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_42 = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_9 = torch.nn.functional.silu(batch_norm_42, inplace=True)
        batch_norm_42 = None
        conv2d_43 = torch.conv2d(
            x_9,
            l_self_modules_model_modules_10_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = (
            l_self_modules_model_modules_10_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_43 = torch.nn.functional.batch_norm(
            conv2d_43,
            l_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_43 = (
            l_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_43 = torch.nn.functional.silu(batch_norm_43, inplace=True)
        batch_norm_43 = None
        split = silu_43.split((256, 256), dim=1)
        silu_43 = None
        a = split[0]
        b = split[1]
        split = None
        conv2d_44 = torch.conv2d(
            b,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        qkv = torch.nn.functional.batch_norm(
            conv2d_44,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_44 = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
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
        conv2d_45 = torch.conv2d(
            reshape,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        reshape = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_45 = torch.nn.functional.batch_norm(
            conv2d_45,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_45 = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_10 = view_1 + batch_norm_45
        view_1 = batch_norm_45 = None
        conv2d_46 = torch.conv2d(
            x_10,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            conv2d_46,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_46 = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_12 = b + x_11
        b = x_11 = None
        conv2d_47 = torch.conv2d(
            x_12,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_47 = torch.nn.functional.batch_norm(
            conv2d_47,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_47 = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_ = (None)
        input_9 = torch.nn.functional.silu(batch_norm_47, inplace=True)
        batch_norm_47 = None
        conv2d_48 = torch.conv2d(
            input_9,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_9 = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_ = (None)
        input_10 = torch.nn.functional.batch_norm(
            conv2d_48,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_48 = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_ = (None)
        x_13 = x_12 + input_10
        x_12 = input_10 = None
        cat_9 = torch.cat((a, x_13), 1)
        a = x_13 = None
        conv2d_49 = torch.conv2d(
            cat_9,
            l_self_modules_model_modules_10_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_9 = (
            l_self_modules_model_modules_10_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_49 = torch.nn.functional.batch_norm(
            conv2d_49,
            l_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_49 = (
            l_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_14 = torch.nn.functional.silu(batch_norm_49, inplace=True)
        batch_norm_49 = None
        x_15 = torch.nn.functional.interpolate(
            x_14, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        x_16 = torch.cat([x_15, x_6], 1)
        x_15 = x_6 = None
        conv2d_50 = torch.conv2d(
            x_16,
            l_self_modules_model_modules_13_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = (
            l_self_modules_model_modules_13_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_50 = torch.nn.functional.batch_norm(
            conv2d_50,
            l_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_50 = (
            l_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_46 = torch.nn.functional.silu(batch_norm_50, inplace=True)
        batch_norm_50 = None
        chunk_4 = silu_46.chunk(2, 1)
        silu_46 = None
        getitem_13 = chunk_4[0]
        getitem_14 = chunk_4[1]
        chunk_4 = None
        conv2d_51 = torch.conv2d(
            getitem_14,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_51 = torch.nn.functional.batch_norm(
            conv2d_51,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_51 = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_47 = torch.nn.functional.silu(batch_norm_51, inplace=True)
        batch_norm_51 = None
        conv2d_52 = torch.conv2d(
            silu_47,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_52 = torch.nn.functional.batch_norm(
            conv2d_52,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_52 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_48 = torch.nn.functional.silu(batch_norm_52, inplace=True)
        batch_norm_52 = None
        conv2d_53 = torch.conv2d(
            silu_48,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_48 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_53 = torch.nn.functional.batch_norm(
            conv2d_53,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_53 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_49 = torch.nn.functional.silu(batch_norm_53, inplace=True)
        batch_norm_53 = None
        input_11 = silu_47 + silu_49
        silu_47 = silu_49 = None
        conv2d_54 = torch.conv2d(
            input_11,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_54 = torch.nn.functional.batch_norm(
            conv2d_54,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_54 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_50 = torch.nn.functional.silu(batch_norm_54, inplace=True)
        batch_norm_54 = None
        conv2d_55 = torch.conv2d(
            silu_50,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_50 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_55 = torch.nn.functional.batch_norm(
            conv2d_55,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_55 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_51 = torch.nn.functional.silu(batch_norm_55, inplace=True)
        batch_norm_55 = None
        input_12 = input_11 + silu_51
        input_11 = silu_51 = None
        conv2d_56 = torch.conv2d(
            getitem_14,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_56 = torch.nn.functional.batch_norm(
            conv2d_56,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_56 = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_52 = torch.nn.functional.silu(batch_norm_56, inplace=True)
        batch_norm_56 = None
        cat_11 = torch.cat((input_12, silu_52), 1)
        input_12 = silu_52 = None
        conv2d_57 = torch.conv2d(
            cat_11,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_11 = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_57 = torch.nn.functional.batch_norm(
            conv2d_57,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_57 = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_53 = torch.nn.functional.silu(batch_norm_57, inplace=True)
        batch_norm_57 = None
        cat_12 = torch.cat([getitem_13, getitem_14, silu_53], 1)
        getitem_13 = getitem_14 = silu_53 = None
        conv2d_58 = torch.conv2d(
            cat_12,
            l_self_modules_model_modules_13_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_12 = (
            l_self_modules_model_modules_13_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_58 = torch.nn.functional.batch_norm(
            conv2d_58,
            l_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_58 = (
            l_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_17 = torch.nn.functional.silu(batch_norm_58, inplace=True)
        batch_norm_58 = None
        x_18 = torch.nn.functional.interpolate(
            x_17, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        x_19 = torch.cat([x_18, x_4], 1)
        x_18 = x_4 = None
        conv2d_59 = torch.conv2d(
            x_19,
            l_self_modules_model_modules_16_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = (
            l_self_modules_model_modules_16_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_59 = torch.nn.functional.batch_norm(
            conv2d_59,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_59 = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_55 = torch.nn.functional.silu(batch_norm_59, inplace=True)
        batch_norm_59 = None
        chunk_5 = silu_55.chunk(2, 1)
        silu_55 = None
        getitem_15 = chunk_5[0]
        getitem_16 = chunk_5[1]
        chunk_5 = None
        conv2d_60 = torch.conv2d(
            getitem_16,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_60 = torch.nn.functional.batch_norm(
            conv2d_60,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_60 = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_56 = torch.nn.functional.silu(batch_norm_60, inplace=True)
        batch_norm_60 = None
        conv2d_61 = torch.conv2d(
            silu_56,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_61 = torch.nn.functional.batch_norm(
            conv2d_61,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_61 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_57 = torch.nn.functional.silu(batch_norm_61, inplace=True)
        batch_norm_61 = None
        conv2d_62 = torch.conv2d(
            silu_57,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_57 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_62 = torch.nn.functional.batch_norm(
            conv2d_62,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_62 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_58 = torch.nn.functional.silu(batch_norm_62, inplace=True)
        batch_norm_62 = None
        input_13 = silu_56 + silu_58
        silu_56 = silu_58 = None
        conv2d_63 = torch.conv2d(
            input_13,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_63 = torch.nn.functional.batch_norm(
            conv2d_63,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_63 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_59 = torch.nn.functional.silu(batch_norm_63, inplace=True)
        batch_norm_63 = None
        conv2d_64 = torch.conv2d(
            silu_59,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_59 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_64 = torch.nn.functional.batch_norm(
            conv2d_64,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_64 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_60 = torch.nn.functional.silu(batch_norm_64, inplace=True)
        batch_norm_64 = None
        input_14 = input_13 + silu_60
        input_13 = silu_60 = None
        conv2d_65 = torch.conv2d(
            getitem_16,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_65 = torch.nn.functional.batch_norm(
            conv2d_65,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_65 = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_61 = torch.nn.functional.silu(batch_norm_65, inplace=True)
        batch_norm_65 = None
        cat_14 = torch.cat((input_14, silu_61), 1)
        input_14 = silu_61 = None
        conv2d_66 = torch.conv2d(
            cat_14,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_14 = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_66 = torch.nn.functional.batch_norm(
            conv2d_66,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_66 = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_62 = torch.nn.functional.silu(batch_norm_66, inplace=True)
        batch_norm_66 = None
        cat_15 = torch.cat([getitem_15, getitem_16, silu_62], 1)
        getitem_15 = getitem_16 = silu_62 = None
        conv2d_67 = torch.conv2d(
            cat_15,
            l_self_modules_model_modules_16_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_15 = (
            l_self_modules_model_modules_16_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_67 = torch.nn.functional.batch_norm(
            conv2d_67,
            l_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_67 = (
            l_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_20 = torch.nn.functional.silu(batch_norm_67, inplace=True)
        batch_norm_67 = None
        conv2d_68 = torch.conv2d(
            x_20,
            l_self_modules_model_modules_17_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_17_modules_conv_parameters_weight_ = None
        batch_norm_68 = torch.nn.functional.batch_norm(
            conv2d_68,
            l_self_modules_model_modules_17_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_17_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_17_modules_bn_parameters_weight_,
            l_self_modules_model_modules_17_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_68 = (
            l_self_modules_model_modules_17_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_17_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_17_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_17_modules_bn_parameters_bias_ = None
        x_21 = torch.nn.functional.silu(batch_norm_68, inplace=True)
        batch_norm_68 = None
        x_22 = torch.cat([x_21, x_17], 1)
        x_21 = x_17 = None
        conv2d_69 = torch.conv2d(
            x_22,
            l_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = (
            l_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_69 = torch.nn.functional.batch_norm(
            conv2d_69,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_69 = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_65 = torch.nn.functional.silu(batch_norm_69, inplace=True)
        batch_norm_69 = None
        chunk_6 = silu_65.chunk(2, 1)
        silu_65 = None
        getitem_17 = chunk_6[0]
        getitem_18 = chunk_6[1]
        chunk_6 = None
        conv2d_70 = torch.conv2d(
            getitem_18,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_70 = torch.nn.functional.batch_norm(
            conv2d_70,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_70 = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_66 = torch.nn.functional.silu(batch_norm_70, inplace=True)
        batch_norm_70 = None
        conv2d_71 = torch.conv2d(
            silu_66,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_71 = torch.nn.functional.batch_norm(
            conv2d_71,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_71 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_67 = torch.nn.functional.silu(batch_norm_71, inplace=True)
        batch_norm_71 = None
        conv2d_72 = torch.conv2d(
            silu_67,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_67 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_72 = torch.nn.functional.batch_norm(
            conv2d_72,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_72 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_68 = torch.nn.functional.silu(batch_norm_72, inplace=True)
        batch_norm_72 = None
        input_15 = silu_66 + silu_68
        silu_66 = silu_68 = None
        conv2d_73 = torch.conv2d(
            input_15,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_73 = torch.nn.functional.batch_norm(
            conv2d_73,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_73 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_69 = torch.nn.functional.silu(batch_norm_73, inplace=True)
        batch_norm_73 = None
        conv2d_74 = torch.conv2d(
            silu_69,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_69 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_74 = torch.nn.functional.batch_norm(
            conv2d_74,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_74 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_70 = torch.nn.functional.silu(batch_norm_74, inplace=True)
        batch_norm_74 = None
        input_16 = input_15 + silu_70
        input_15 = silu_70 = None
        conv2d_75 = torch.conv2d(
            getitem_18,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_75 = torch.nn.functional.batch_norm(
            conv2d_75,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_75 = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_71 = torch.nn.functional.silu(batch_norm_75, inplace=True)
        batch_norm_75 = None
        cat_17 = torch.cat((input_16, silu_71), 1)
        input_16 = silu_71 = None
        conv2d_76 = torch.conv2d(
            cat_17,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_17 = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_76 = torch.nn.functional.batch_norm(
            conv2d_76,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_76 = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_72 = torch.nn.functional.silu(batch_norm_76, inplace=True)
        batch_norm_76 = None
        cat_18 = torch.cat([getitem_17, getitem_18, silu_72], 1)
        getitem_17 = getitem_18 = silu_72 = None
        conv2d_77 = torch.conv2d(
            cat_18,
            l_self_modules_model_modules_19_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_18 = (
            l_self_modules_model_modules_19_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_77 = torch.nn.functional.batch_norm(
            conv2d_77,
            l_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_77 = (
            l_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_23 = torch.nn.functional.silu(batch_norm_77, inplace=True)
        batch_norm_77 = None
        conv2d_78 = torch.conv2d(
            x_23,
            l_self_modules_model_modules_20_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_20_modules_conv_parameters_weight_ = None
        batch_norm_78 = torch.nn.functional.batch_norm(
            conv2d_78,
            l_self_modules_model_modules_20_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_20_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_20_modules_bn_parameters_weight_,
            l_self_modules_model_modules_20_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_78 = (
            l_self_modules_model_modules_20_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_20_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_20_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_20_modules_bn_parameters_bias_ = None
        x_24 = torch.nn.functional.silu(batch_norm_78, inplace=True)
        batch_norm_78 = None
        x_25 = torch.cat([x_24, x_14], 1)
        x_24 = x_14 = None
        conv2d_79 = torch.conv2d(
            x_25,
            l_self_modules_model_modules_22_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = (
            l_self_modules_model_modules_22_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_79 = torch.nn.functional.batch_norm(
            conv2d_79,
            l_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_79 = (
            l_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_75 = torch.nn.functional.silu(batch_norm_79, inplace=True)
        batch_norm_79 = None
        chunk_7 = silu_75.chunk(2, 1)
        silu_75 = None
        getitem_19 = chunk_7[0]
        getitem_20 = chunk_7[1]
        chunk_7 = None
        conv2d_80 = torch.conv2d(
            getitem_20,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_80 = torch.nn.functional.batch_norm(
            conv2d_80,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_80 = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_76 = torch.nn.functional.silu(batch_norm_80, inplace=True)
        batch_norm_80 = None
        conv2d_81 = torch.conv2d(
            silu_76,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_81 = torch.nn.functional.batch_norm(
            conv2d_81,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_81 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_77 = torch.nn.functional.silu(batch_norm_81, inplace=True)
        batch_norm_81 = None
        conv2d_82 = torch.conv2d(
            silu_77,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_77 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_82 = torch.nn.functional.batch_norm(
            conv2d_82,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_82 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_78 = torch.nn.functional.silu(batch_norm_82, inplace=True)
        batch_norm_82 = None
        input_17 = silu_76 + silu_78
        silu_76 = silu_78 = None
        conv2d_83 = torch.conv2d(
            input_17,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_83 = torch.nn.functional.batch_norm(
            conv2d_83,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_83 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_79 = torch.nn.functional.silu(batch_norm_83, inplace=True)
        batch_norm_83 = None
        conv2d_84 = torch.conv2d(
            silu_79,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_79 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_84 = torch.nn.functional.batch_norm(
            conv2d_84,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_84 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_80 = torch.nn.functional.silu(batch_norm_84, inplace=True)
        batch_norm_84 = None
        input_18 = input_17 + silu_80
        input_17 = silu_80 = None
        conv2d_85 = torch.conv2d(
            getitem_20,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_85 = torch.nn.functional.batch_norm(
            conv2d_85,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_85 = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_81 = torch.nn.functional.silu(batch_norm_85, inplace=True)
        batch_norm_85 = None
        cat_20 = torch.cat((input_18, silu_81), 1)
        input_18 = silu_81 = None
        conv2d_86 = torch.conv2d(
            cat_20,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_20 = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_86 = torch.nn.functional.batch_norm(
            conv2d_86,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_86 = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_82 = torch.nn.functional.silu(batch_norm_86, inplace=True)
        batch_norm_86 = None
        cat_21 = torch.cat([getitem_19, getitem_20, silu_82], 1)
        getitem_19 = getitem_20 = silu_82 = None
        conv2d_87 = torch.conv2d(
            cat_21,
            l_self_modules_model_modules_22_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_21 = (
            l_self_modules_model_modules_22_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_87 = torch.nn.functional.batch_norm(
            conv2d_87,
            l_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_87 = (
            l_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_26 = torch.nn.functional.silu(batch_norm_87, inplace=True)
        batch_norm_87 = None
        conv2d_88 = torch.conv2d(
            x_20,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_88 = torch.nn.functional.batch_norm(
            conv2d_88,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_88 = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_19 = torch.nn.functional.silu(batch_norm_88, inplace=True)
        batch_norm_88 = None
        conv2d_89 = torch.conv2d(
            input_19,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_19 = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_89 = torch.nn.functional.batch_norm(
            conv2d_89,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_89 = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_20 = torch.nn.functional.silu(batch_norm_89, inplace=True)
        batch_norm_89 = None
        input_21 = torch.conv2d(
            input_20,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_20 = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_bias_ = (None)
        conv2d_91 = torch.conv2d(
            x_20,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_20 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_90 = torch.nn.functional.batch_norm(
            conv2d_91,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_91 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_22 = torch.nn.functional.silu(batch_norm_90, inplace=True)
        batch_norm_90 = None
        conv2d_92 = torch.conv2d(
            input_22,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_22 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_91 = torch.nn.functional.batch_norm(
            conv2d_92,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_92 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_23 = torch.nn.functional.silu(batch_norm_91, inplace=True)
        batch_norm_91 = None
        conv2d_93 = torch.conv2d(
            input_23,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_23 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_92 = torch.nn.functional.batch_norm(
            conv2d_93,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_93 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_24 = torch.nn.functional.silu(batch_norm_92, inplace=True)
        batch_norm_92 = None
        conv2d_94 = torch.conv2d(
            input_24,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_24 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_93 = torch.nn.functional.batch_norm(
            conv2d_94,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_94 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_25 = torch.nn.functional.silu(batch_norm_93, inplace=True)
        batch_norm_93 = None
        input_26 = torch.conv2d(
            input_25,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_25 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_bias_ = (None)
        xi = torch.cat((input_21, input_26), 1)
        input_21 = input_26 = None
        conv2d_96 = torch.conv2d(
            x_23,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_94 = torch.nn.functional.batch_norm(
            conv2d_96,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_96 = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_27 = torch.nn.functional.silu(batch_norm_94, inplace=True)
        batch_norm_94 = None
        conv2d_97 = torch.conv2d(
            input_27,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_27 = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_95 = torch.nn.functional.batch_norm(
            conv2d_97,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_97 = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_28 = torch.nn.functional.silu(batch_norm_95, inplace=True)
        batch_norm_95 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_28 = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_bias_ = (None)
        conv2d_99 = torch.conv2d(
            x_23,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_23 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_96 = torch.nn.functional.batch_norm(
            conv2d_99,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_99 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_30 = torch.nn.functional.silu(batch_norm_96, inplace=True)
        batch_norm_96 = None
        conv2d_100 = torch.conv2d(
            input_30,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_30 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_97 = torch.nn.functional.batch_norm(
            conv2d_100,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_100 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_31 = torch.nn.functional.silu(batch_norm_97, inplace=True)
        batch_norm_97 = None
        conv2d_101 = torch.conv2d(
            input_31,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_31 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_98 = torch.nn.functional.batch_norm(
            conv2d_101,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_101 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_32 = torch.nn.functional.silu(batch_norm_98, inplace=True)
        batch_norm_98 = None
        conv2d_102 = torch.conv2d(
            input_32,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_32 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_99 = torch.nn.functional.batch_norm(
            conv2d_102,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_102 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_33 = torch.nn.functional.silu(batch_norm_99, inplace=True)
        batch_norm_99 = None
        input_34 = torch.conv2d(
            input_33,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_33 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_bias_ = (None)
        xi_1 = torch.cat((input_29, input_34), 1)
        input_29 = input_34 = None
        conv2d_104 = torch.conv2d(
            x_26,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_100 = torch.nn.functional.batch_norm(
            conv2d_104,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_104 = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        input_35 = torch.nn.functional.silu(batch_norm_100, inplace=True)
        batch_norm_100 = None
        conv2d_105 = torch.conv2d(
            input_35,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_35 = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_101 = torch.nn.functional.batch_norm(
            conv2d_105,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_105 = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        input_36 = torch.nn.functional.silu(batch_norm_101, inplace=True)
        batch_norm_101 = None
        input_37 = torch.conv2d(
            input_36,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_36 = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_bias_ = (None)
        conv2d_107 = torch.conv2d(
            x_26,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_26 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_102 = torch.nn.functional.batch_norm(
            conv2d_107,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_107 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_38 = torch.nn.functional.silu(batch_norm_102, inplace=True)
        batch_norm_102 = None
        conv2d_108 = torch.conv2d(
            input_38,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_38 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_103 = torch.nn.functional.batch_norm(
            conv2d_108,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_108 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_39 = torch.nn.functional.silu(batch_norm_103, inplace=True)
        batch_norm_103 = None
        conv2d_109 = torch.conv2d(
            input_39,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_39 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_104 = torch.nn.functional.batch_norm(
            conv2d_109,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_109 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_40 = torch.nn.functional.silu(batch_norm_104, inplace=True)
        batch_norm_104 = None
        conv2d_110 = torch.conv2d(
            input_40,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_40 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_105 = torch.nn.functional.batch_norm(
            conv2d_110,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_110 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_41 = torch.nn.functional.silu(batch_norm_105, inplace=True)
        batch_norm_105 = None
        input_42 = torch.conv2d(
            input_41,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_41 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_bias_ = (None)
        xi_2 = torch.cat((input_37, input_42), 1)
        input_37 = input_42 = None
        view_2 = xi.view(1, 144, -1)
        view_3 = xi_1.view(1, 144, -1)
        view_4 = xi_2.view(1, 144, -1)
        x_cat = torch.cat([view_2, view_3, view_4], 2)
        view_2 = view_3 = view_4 = None
        x_27 = l_self_modules_model_modules_23_stride[0]
        x_28 = l_self_modules_model_modules_23_stride[1]
        x_29 = l_self_modules_model_modules_23_stride[2]
        l_self_modules_model_modules_23_stride = None
        arange = torch.arange(
            end=80, device=device(type="cuda", index=0), dtype=torch.float32
        )
        sx = arange + 0.5
        arange = None
        arange_1 = torch.arange(
            end=80, device=device(type="cuda", index=0), dtype=torch.float32
        )
        sy = arange_1 + 0.5
        arange_1 = None
        meshgrid = torch.functional.meshgrid(sy, sx, indexing="ij")
        sy = sx = None
        sy_1 = meshgrid[0]
        sx_1 = meshgrid[1]
        meshgrid = None
        stack = torch.stack((sx_1, sy_1), -1)
        sx_1 = sy_1 = None
        view_5 = stack.view(-1, 2)
        stack = None
        _local_scalar_dense = torch.ops.aten._local_scalar_dense(x_27)
        x_27 = None
        full = torch.full(
            (6400, 1),
            _local_scalar_dense,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        _local_scalar_dense = None
        arange_2 = torch.arange(
            end=40, device=device(type="cuda", index=0), dtype=torch.float32
        )
        sx_2 = arange_2 + 0.5
        arange_2 = None
        arange_3 = torch.arange(
            end=40, device=device(type="cuda", index=0), dtype=torch.float32
        )
        sy_2 = arange_3 + 0.5
        arange_3 = None
        meshgrid_1 = torch.functional.meshgrid(sy_2, sx_2, indexing="ij")
        sy_2 = sx_2 = None
        sy_3 = meshgrid_1[0]
        sx_3 = meshgrid_1[1]
        meshgrid_1 = None
        stack_1 = torch.stack((sx_3, sy_3), -1)
        sx_3 = sy_3 = None
        view_6 = stack_1.view(-1, 2)
        stack_1 = None
        _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense(x_28)
        x_28 = None
        full_1 = torch.full(
            (1600, 1),
            _local_scalar_dense_1,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        _local_scalar_dense_1 = None
        arange_4 = torch.arange(
            end=20, device=device(type="cuda", index=0), dtype=torch.float32
        )
        sx_4 = arange_4 + 0.5
        arange_4 = None
        arange_5 = torch.arange(
            end=20, device=device(type="cuda", index=0), dtype=torch.float32
        )
        sy_4 = arange_5 + 0.5
        arange_5 = None
        meshgrid_2 = torch.functional.meshgrid(sy_4, sx_4, indexing="ij")
        sy_4 = sx_4 = None
        sy_5 = meshgrid_2[0]
        sx_5 = meshgrid_2[1]
        meshgrid_2 = None
        stack_2 = torch.stack((sx_5, sy_5), -1)
        sx_5 = sy_5 = None
        view_7 = stack_2.view(-1, 2)
        stack_2 = None
        _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense(x_29)
        x_29 = None
        full_2 = torch.full(
            (400, 1),
            _local_scalar_dense_2,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        _local_scalar_dense_2 = None
        x_30 = torch.cat([view_5, view_6, view_7])
        view_5 = view_6 = view_7 = None
        x_31 = torch.cat([full, full_1, full_2])
        full = full_1 = full_2 = None
        transpose_2 = x_30.transpose(0, 1)
        x_30 = None
        transpose_3 = x_31.transpose(0, 1)
        x_31 = None
        split_2 = x_cat.split((64, 80), 1)
        x_cat = None
        box = split_2[0]
        cls = split_2[1]
        split_2 = None
        view_8 = box.view(1, 4, 16, 8400)
        box = None
        transpose_4 = view_8.transpose(2, 1)
        view_8 = None
        softmax_1 = transpose_4.softmax(1)
        transpose_4 = None
        conv2d_112 = torch.conv2d(
            softmax_1,
            l_self_modules_model_modules_23_modules_dfl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        softmax_1 = (
            l_self_modules_model_modules_23_modules_dfl_modules_conv_parameters_weight_
        ) = None
        view_9 = conv2d_112.view(1, 4, 8400)
        conv2d_112 = None
        unsqueeze = transpose_2.unsqueeze(0)
        chunk_8 = view_9.chunk(2, 1)
        view_9 = None
        lt = chunk_8[0]
        rb = chunk_8[1]
        chunk_8 = None
        x1y1 = unsqueeze - lt
        lt = None
        x2y2 = unsqueeze + rb
        unsqueeze = rb = None
        add_26 = x1y1 + x2y2
        c_xy = add_26 / 2
        add_26 = None
        wh = x2y2 - x1y1
        x2y2 = x1y1 = None
        cat_28 = torch.cat((c_xy, wh), 1)
        c_xy = wh = None
        dbox = cat_28 * transpose_3
        cat_28 = None
        sigmoid = cls.sigmoid()
        cls = None
        y = torch.cat((dbox, sigmoid), 1)
        dbox = sigmoid = None
        return (y, xi, xi_1, xi_2, transpose_3, transpose_2)
