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
        L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_3_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_3_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_5_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_5_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_7_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_7_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_cv5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_cv5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_12_modules_cv4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_12_modules_cv4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_cv4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_cv4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_cv4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_cv4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_21_modules_cv4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_21_modules_cv4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_stride: torch.Tensor,
        L_self_modules_model_modules_22_modules_dfl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_model_modules_2_modules_cv3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv3_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_2_modules_cv4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_2_modules_cv4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_2_modules_cv4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_2_modules_cv4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_2_modules_cv4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_2_modules_cv4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_3_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_3_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_3_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_3_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_3_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_3_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_3_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_3_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_3_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_3_modules_cv1_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv3_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_cv4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_cv4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_cv4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_cv4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_cv4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_cv4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_cv4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_cv4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_5_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_5_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_5_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_5_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_5_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_5_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_5_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_5_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_5_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_5_modules_cv1_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv3_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_cv4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_cv4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_cv4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_cv4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_cv4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_cv4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_cv4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_cv4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_7_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_7_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_7_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_7_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_7_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_7_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_7_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_7_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_7_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_7_modules_cv1_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv3_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_cv4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_cv4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_8_modules_cv4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_8_modules_cv4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_8_modules_cv4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_8_modules_cv4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_cv4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_8_modules_cv4_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_9_modules_cv5_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_cv5_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_cv5_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_9_modules_cv5_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_9_modules_cv5_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_9_modules_cv5_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_9_modules_cv5_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_cv5_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_cv5_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_9_modules_cv5_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_12_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_12_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv3_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_12_modules_cv4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_12_modules_cv4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_12_modules_cv4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_12_modules_cv4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_12_modules_cv4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_12_modules_cv4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_12_modules_cv4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_12_modules_cv4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_12_modules_cv4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_12_modules_cv4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_15_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_15_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv3_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_15_modules_cv4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_15_modules_cv4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_15_modules_cv4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_15_modules_cv4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_15_modules_cv4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_15_modules_cv4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_15_modules_cv4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_15_modules_cv4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_15_modules_cv4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_15_modules_cv4_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_18_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_18_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv3_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_18_modules_cv4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_18_modules_cv4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_18_modules_cv4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_18_modules_cv4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_18_modules_cv4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_18_modules_cv4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_18_modules_cv4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_18_modules_cv4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_18_modules_cv4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_18_modules_cv4_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_21_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_21_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv3_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_21_modules_cv4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_21_modules_cv4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_21_modules_cv4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_21_modules_cv4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_21_modules_cv4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_21_modules_cv4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_21_modules_cv4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_21_modules_cv4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_21_modules_cv4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_21_modules_cv4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_bias_ = L_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_bias_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_bias_ = L_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_bias_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_bias_ = L_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_bias_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_bias_ = L_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_bias_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_weight_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_weight_
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_bias_ = L_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_bias_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_weight_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_weight_
        l_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_bias_ = L_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_bias_
        l_self_modules_model_modules_22_stride = L_self_modules_model_modules_22_stride
        l_self_modules_model_modules_22_modules_dfl_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_22_modules_dfl_modules_conv_parameters_weight_
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
            l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_3 = torch.nn.functional.batch_norm(
            conv2d_3,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_3 = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_
        ) = None
        silu_3 = torch.nn.functional.silu(batch_norm_3, inplace=True)
        batch_norm_3 = None
        conv2d_4 = torch.conv2d(
            getitem_1,
            l_self_modules_model_modules_2_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_cv3_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_4 = torch.nn.functional.batch_norm(
            conv2d_4,
            l_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_4 = (
            l_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_bias_
        ) = None
        silu_4 = torch.nn.functional.silu(batch_norm_4, inplace=True)
        batch_norm_4 = None
        cat = torch.cat([getitem, getitem_1, silu_3, silu_4], 1)
        getitem = getitem_1 = silu_3 = silu_4 = None
        conv2d_5 = torch.conv2d(
            cat,
            l_self_modules_model_modules_2_modules_cv4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = (
            l_self_modules_model_modules_2_modules_cv4_modules_conv_parameters_weight_
        ) = None
        batch_norm_5 = torch.nn.functional.batch_norm(
            conv2d_5,
            l_self_modules_model_modules_2_modules_cv4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_cv4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_cv4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_cv4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_5 = (
            l_self_modules_model_modules_2_modules_cv4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_cv4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_cv4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_2_modules_cv4_modules_bn_parameters_bias_
        ) = None
        x_2 = torch.nn.functional.silu(batch_norm_5, inplace=True)
        batch_norm_5 = None
        x_3 = torch._C._nn.avg_pool2d(x_2, 2, 1, 0, False, True)
        x_2 = None
        conv2d_6 = torch.conv2d(
            x_3,
            l_self_modules_model_modules_3_modules_cv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_3 = (
            l_self_modules_model_modules_3_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_6 = torch.nn.functional.batch_norm(
            conv2d_6,
            l_self_modules_model_modules_3_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_3_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_3_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_3_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_6 = (
            l_self_modules_model_modules_3_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_3_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_3_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_3_modules_cv1_modules_bn_parameters_bias_
        ) = None
        x_4 = torch.nn.functional.silu(batch_norm_6, inplace=True)
        batch_norm_6 = None
        conv2d_7 = torch.conv2d(
            x_4,
            l_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_4 = (
            l_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_7 = torch.nn.functional.batch_norm(
            conv2d_7,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_7 = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_7 = torch.nn.functional.silu(batch_norm_7, inplace=True)
        batch_norm_7 = None
        chunk_1 = silu_7.chunk(2, 1)
        silu_7 = None
        getitem_2 = chunk_1[0]
        getitem_3 = chunk_1[1]
        chunk_1 = None
        conv2d_8 = torch.conv2d(
            getitem_3,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_8 = torch.nn.functional.batch_norm(
            conv2d_8,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_8 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_8 = torch.nn.functional.silu(batch_norm_8, inplace=True)
        batch_norm_8 = None
        conv2d_9 = torch.conv2d(
            silu_8,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_9 = torch.nn.functional.batch_norm(
            conv2d_9,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_9 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_10 = torch.conv2d(
            silu_8,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_10 = torch.nn.functional.batch_norm(
            conv2d_10,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_10 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add = batch_norm_9 + batch_norm_10
        batch_norm_9 = batch_norm_10 = None
        add_1 = add + 0
        add = None
        silu_9 = torch.nn.functional.silu(add_1, inplace=True)
        add_1 = None
        conv2d_11 = torch.conv2d(
            silu_9,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_9 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_11 = torch.nn.functional.batch_norm(
            conv2d_11,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_11 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_10 = torch.nn.functional.silu(batch_norm_11, inplace=True)
        batch_norm_11 = None
        input_1 = silu_8 + silu_10
        silu_8 = silu_10 = None
        conv2d_12 = torch.conv2d(
            input_1,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_12 = torch.nn.functional.batch_norm(
            conv2d_12,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_12 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_13 = torch.conv2d(
            input_1,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_13 = torch.nn.functional.batch_norm(
            conv2d_13,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_13 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_3 = batch_norm_12 + batch_norm_13
        batch_norm_12 = batch_norm_13 = None
        add_4 = add_3 + 0
        add_3 = None
        silu_11 = torch.nn.functional.silu(add_4, inplace=True)
        add_4 = None
        conv2d_14 = torch.conv2d(
            silu_11,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_11 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_14 = torch.nn.functional.batch_norm(
            conv2d_14,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_14 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_12 = torch.nn.functional.silu(batch_norm_14, inplace=True)
        batch_norm_14 = None
        input_2 = input_1 + silu_12
        input_1 = silu_12 = None
        conv2d_15 = torch.conv2d(
            input_2,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_15 = torch.nn.functional.batch_norm(
            conv2d_15,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_15 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_16 = torch.conv2d(
            input_2,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_16 = torch.nn.functional.batch_norm(
            conv2d_16,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_16 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_6 = batch_norm_15 + batch_norm_16
        batch_norm_15 = batch_norm_16 = None
        add_7 = add_6 + 0
        add_6 = None
        silu_13 = torch.nn.functional.silu(add_7, inplace=True)
        add_7 = None
        conv2d_17 = torch.conv2d(
            silu_13,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_13 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_17 = torch.nn.functional.batch_norm(
            conv2d_17,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_17 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_14 = torch.nn.functional.silu(batch_norm_17, inplace=True)
        batch_norm_17 = None
        input_3 = input_2 + silu_14
        input_2 = silu_14 = None
        conv2d_18 = torch.conv2d(
            getitem_3,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_18 = torch.nn.functional.batch_norm(
            conv2d_18,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_18 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_15 = torch.nn.functional.silu(batch_norm_18, inplace=True)
        batch_norm_18 = None
        cat_1 = torch.cat((input_3, silu_15), 1)
        input_3 = silu_15 = None
        conv2d_19 = torch.conv2d(
            cat_1,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_19 = torch.nn.functional.batch_norm(
            conv2d_19,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_19 = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_4 = torch.nn.functional.silu(batch_norm_19, inplace=True)
        batch_norm_19 = None
        conv2d_20 = torch.conv2d(
            input_4,
            l_self_modules_model_modules_4_modules_cv2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_model_modules_4_modules_cv2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_20 = torch.nn.functional.batch_norm(
            conv2d_20,
            l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_20 = l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv2_modules_1_modules_bn_parameters_bias_ = (None)
        input_5 = torch.nn.functional.silu(batch_norm_20, inplace=True)
        batch_norm_20 = None
        conv2d_21 = torch.conv2d(
            getitem_3,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_21 = torch.nn.functional.batch_norm(
            conv2d_21,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_21 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_18 = torch.nn.functional.silu(batch_norm_21, inplace=True)
        batch_norm_21 = None
        conv2d_22 = torch.conv2d(
            silu_18,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_22 = torch.nn.functional.batch_norm(
            conv2d_22,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_22 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_23 = torch.conv2d(
            silu_18,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_23 = torch.nn.functional.batch_norm(
            conv2d_23,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_23 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_9 = batch_norm_22 + batch_norm_23
        batch_norm_22 = batch_norm_23 = None
        add_10 = add_9 + 0
        add_9 = None
        silu_19 = torch.nn.functional.silu(add_10, inplace=True)
        add_10 = None
        conv2d_24 = torch.conv2d(
            silu_19,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_19 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_24 = torch.nn.functional.batch_norm(
            conv2d_24,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_24 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_20 = torch.nn.functional.silu(batch_norm_24, inplace=True)
        batch_norm_24 = None
        input_6 = silu_18 + silu_20
        silu_18 = silu_20 = None
        conv2d_25 = torch.conv2d(
            input_6,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_25 = torch.nn.functional.batch_norm(
            conv2d_25,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_25 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_26 = torch.conv2d(
            input_6,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_26 = torch.nn.functional.batch_norm(
            conv2d_26,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_26 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_12 = batch_norm_25 + batch_norm_26
        batch_norm_25 = batch_norm_26 = None
        add_13 = add_12 + 0
        add_12 = None
        silu_21 = torch.nn.functional.silu(add_13, inplace=True)
        add_13 = None
        conv2d_27 = torch.conv2d(
            silu_21,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_21 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_27 = torch.nn.functional.batch_norm(
            conv2d_27,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_27 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_22 = torch.nn.functional.silu(batch_norm_27, inplace=True)
        batch_norm_27 = None
        input_7 = input_6 + silu_22
        input_6 = silu_22 = None
        conv2d_28 = torch.conv2d(
            input_7,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_28 = torch.nn.functional.batch_norm(
            conv2d_28,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_28 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_29 = torch.conv2d(
            input_7,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_29 = torch.nn.functional.batch_norm(
            conv2d_29,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_29 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_15 = batch_norm_28 + batch_norm_29
        batch_norm_28 = batch_norm_29 = None
        add_16 = add_15 + 0
        add_15 = None
        silu_23 = torch.nn.functional.silu(add_16, inplace=True)
        add_16 = None
        conv2d_30 = torch.conv2d(
            silu_23,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_23 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_30 = torch.nn.functional.batch_norm(
            conv2d_30,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_30 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_24 = torch.nn.functional.silu(batch_norm_30, inplace=True)
        batch_norm_30 = None
        input_8 = input_7 + silu_24
        input_7 = silu_24 = None
        conv2d_31 = torch.conv2d(
            getitem_3,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_31 = torch.nn.functional.batch_norm(
            conv2d_31,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_31 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_25 = torch.nn.functional.silu(batch_norm_31, inplace=True)
        batch_norm_31 = None
        cat_2 = torch.cat((input_8, silu_25), 1)
        input_8 = silu_25 = None
        conv2d_32 = torch.conv2d(
            cat_2,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_32 = torch.nn.functional.batch_norm(
            conv2d_32,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_32 = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_9 = torch.nn.functional.silu(batch_norm_32, inplace=True)
        batch_norm_32 = None
        conv2d_33 = torch.conv2d(
            input_9,
            l_self_modules_model_modules_4_modules_cv3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_9 = l_self_modules_model_modules_4_modules_cv3_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_33 = torch.nn.functional.batch_norm(
            conv2d_33,
            l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_33 = l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_cv3_modules_1_modules_bn_parameters_bias_ = (None)
        input_10 = torch.nn.functional.silu(batch_norm_33, inplace=True)
        batch_norm_33 = None
        cat_3 = torch.cat([getitem_2, getitem_3, input_5, input_10], 1)
        getitem_2 = getitem_3 = input_5 = input_10 = None
        conv2d_34 = torch.conv2d(
            cat_3,
            l_self_modules_model_modules_4_modules_cv4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = (
            l_self_modules_model_modules_4_modules_cv4_modules_conv_parameters_weight_
        ) = None
        batch_norm_34 = torch.nn.functional.batch_norm(
            conv2d_34,
            l_self_modules_model_modules_4_modules_cv4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_34 = (
            l_self_modules_model_modules_4_modules_cv4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_cv4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_cv4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_4_modules_cv4_modules_bn_parameters_bias_
        ) = None
        x_5 = torch.nn.functional.silu(batch_norm_34, inplace=True)
        batch_norm_34 = None
        x_6 = torch._C._nn.avg_pool2d(x_5, 2, 1, 0, False, True)
        conv2d_35 = torch.conv2d(
            x_6,
            l_self_modules_model_modules_5_modules_cv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_6 = (
            l_self_modules_model_modules_5_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_35 = torch.nn.functional.batch_norm(
            conv2d_35,
            l_self_modules_model_modules_5_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_5_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_5_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_5_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_35 = (
            l_self_modules_model_modules_5_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_5_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_5_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_5_modules_cv1_modules_bn_parameters_bias_
        ) = None
        x_7 = torch.nn.functional.silu(batch_norm_35, inplace=True)
        batch_norm_35 = None
        conv2d_36 = torch.conv2d(
            x_7,
            l_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = (
            l_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_36 = torch.nn.functional.batch_norm(
            conv2d_36,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
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
        silu_30 = torch.nn.functional.silu(batch_norm_36, inplace=True)
        batch_norm_36 = None
        chunk_2 = silu_30.chunk(2, 1)
        silu_30 = None
        getitem_4 = chunk_2[0]
        getitem_5 = chunk_2[1]
        chunk_2 = None
        conv2d_37 = torch.conv2d(
            getitem_5,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_37 = torch.nn.functional.batch_norm(
            conv2d_37,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_37 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_31 = torch.nn.functional.silu(batch_norm_37, inplace=True)
        batch_norm_37 = None
        conv2d_38 = torch.conv2d(
            silu_31,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_38 = torch.nn.functional.batch_norm(
            conv2d_38,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_38 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_39 = torch.conv2d(
            silu_31,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_39 = torch.nn.functional.batch_norm(
            conv2d_39,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_39 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_18 = batch_norm_38 + batch_norm_39
        batch_norm_38 = batch_norm_39 = None
        add_19 = add_18 + 0
        add_18 = None
        silu_32 = torch.nn.functional.silu(add_19, inplace=True)
        add_19 = None
        conv2d_40 = torch.conv2d(
            silu_32,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_32 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_40 = torch.nn.functional.batch_norm(
            conv2d_40,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_40 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_33 = torch.nn.functional.silu(batch_norm_40, inplace=True)
        batch_norm_40 = None
        input_11 = silu_31 + silu_33
        silu_31 = silu_33 = None
        conv2d_41 = torch.conv2d(
            input_11,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_41 = torch.nn.functional.batch_norm(
            conv2d_41,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_41 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_42 = torch.conv2d(
            input_11,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_42 = torch.nn.functional.batch_norm(
            conv2d_42,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_42 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_21 = batch_norm_41 + batch_norm_42
        batch_norm_41 = batch_norm_42 = None
        add_22 = add_21 + 0
        add_21 = None
        silu_34 = torch.nn.functional.silu(add_22, inplace=True)
        add_22 = None
        conv2d_43 = torch.conv2d(
            silu_34,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_34 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_43 = torch.nn.functional.batch_norm(
            conv2d_43,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_43 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_35 = torch.nn.functional.silu(batch_norm_43, inplace=True)
        batch_norm_43 = None
        input_12 = input_11 + silu_35
        input_11 = silu_35 = None
        conv2d_44 = torch.conv2d(
            input_12,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_44 = torch.nn.functional.batch_norm(
            conv2d_44,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_44 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_45 = torch.conv2d(
            input_12,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_45 = torch.nn.functional.batch_norm(
            conv2d_45,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_45 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_24 = batch_norm_44 + batch_norm_45
        batch_norm_44 = batch_norm_45 = None
        add_25 = add_24 + 0
        add_24 = None
        silu_36 = torch.nn.functional.silu(add_25, inplace=True)
        add_25 = None
        conv2d_46 = torch.conv2d(
            silu_36,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_36 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_46 = torch.nn.functional.batch_norm(
            conv2d_46,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_46 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_37 = torch.nn.functional.silu(batch_norm_46, inplace=True)
        batch_norm_46 = None
        input_13 = input_12 + silu_37
        input_12 = silu_37 = None
        conv2d_47 = torch.conv2d(
            getitem_5,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_47 = torch.nn.functional.batch_norm(
            conv2d_47,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_47 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_38 = torch.nn.functional.silu(batch_norm_47, inplace=True)
        batch_norm_47 = None
        cat_4 = torch.cat((input_13, silu_38), 1)
        input_13 = silu_38 = None
        conv2d_48 = torch.conv2d(
            cat_4,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_4 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_48 = torch.nn.functional.batch_norm(
            conv2d_48,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_48 = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_14 = torch.nn.functional.silu(batch_norm_48, inplace=True)
        batch_norm_48 = None
        conv2d_49 = torch.conv2d(
            input_14,
            l_self_modules_model_modules_6_modules_cv2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_14 = l_self_modules_model_modules_6_modules_cv2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_49 = torch.nn.functional.batch_norm(
            conv2d_49,
            l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_49 = l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv2_modules_1_modules_bn_parameters_bias_ = (None)
        input_15 = torch.nn.functional.silu(batch_norm_49, inplace=True)
        batch_norm_49 = None
        conv2d_50 = torch.conv2d(
            getitem_5,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_50 = torch.nn.functional.batch_norm(
            conv2d_50,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_50 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_41 = torch.nn.functional.silu(batch_norm_50, inplace=True)
        batch_norm_50 = None
        conv2d_51 = torch.conv2d(
            silu_41,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_51 = torch.nn.functional.batch_norm(
            conv2d_51,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_51 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_52 = torch.conv2d(
            silu_41,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_52 = torch.nn.functional.batch_norm(
            conv2d_52,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_52 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_27 = batch_norm_51 + batch_norm_52
        batch_norm_51 = batch_norm_52 = None
        add_28 = add_27 + 0
        add_27 = None
        silu_42 = torch.nn.functional.silu(add_28, inplace=True)
        add_28 = None
        conv2d_53 = torch.conv2d(
            silu_42,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_42 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_53 = torch.nn.functional.batch_norm(
            conv2d_53,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_53 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_43 = torch.nn.functional.silu(batch_norm_53, inplace=True)
        batch_norm_53 = None
        input_16 = silu_41 + silu_43
        silu_41 = silu_43 = None
        conv2d_54 = torch.conv2d(
            input_16,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_54 = torch.nn.functional.batch_norm(
            conv2d_54,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_54 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_55 = torch.conv2d(
            input_16,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_55 = torch.nn.functional.batch_norm(
            conv2d_55,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_55 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_30 = batch_norm_54 + batch_norm_55
        batch_norm_54 = batch_norm_55 = None
        add_31 = add_30 + 0
        add_30 = None
        silu_44 = torch.nn.functional.silu(add_31, inplace=True)
        add_31 = None
        conv2d_56 = torch.conv2d(
            silu_44,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_44 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_56 = torch.nn.functional.batch_norm(
            conv2d_56,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_56 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_45 = torch.nn.functional.silu(batch_norm_56, inplace=True)
        batch_norm_56 = None
        input_17 = input_16 + silu_45
        input_16 = silu_45 = None
        conv2d_57 = torch.conv2d(
            input_17,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_57 = torch.nn.functional.batch_norm(
            conv2d_57,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_57 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_58 = torch.conv2d(
            input_17,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_58 = torch.nn.functional.batch_norm(
            conv2d_58,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_58 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_33 = batch_norm_57 + batch_norm_58
        batch_norm_57 = batch_norm_58 = None
        add_34 = add_33 + 0
        add_33 = None
        silu_46 = torch.nn.functional.silu(add_34, inplace=True)
        add_34 = None
        conv2d_59 = torch.conv2d(
            silu_46,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_46 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_59 = torch.nn.functional.batch_norm(
            conv2d_59,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_59 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_47 = torch.nn.functional.silu(batch_norm_59, inplace=True)
        batch_norm_59 = None
        input_18 = input_17 + silu_47
        input_17 = silu_47 = None
        conv2d_60 = torch.conv2d(
            getitem_5,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_60 = torch.nn.functional.batch_norm(
            conv2d_60,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_60 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_48 = torch.nn.functional.silu(batch_norm_60, inplace=True)
        batch_norm_60 = None
        cat_5 = torch.cat((input_18, silu_48), 1)
        input_18 = silu_48 = None
        conv2d_61 = torch.conv2d(
            cat_5,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_5 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_61 = torch.nn.functional.batch_norm(
            conv2d_61,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_61 = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_19 = torch.nn.functional.silu(batch_norm_61, inplace=True)
        batch_norm_61 = None
        conv2d_62 = torch.conv2d(
            input_19,
            l_self_modules_model_modules_6_modules_cv3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_19 = l_self_modules_model_modules_6_modules_cv3_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_62 = torch.nn.functional.batch_norm(
            conv2d_62,
            l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_62 = l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_cv3_modules_1_modules_bn_parameters_bias_ = (None)
        input_20 = torch.nn.functional.silu(batch_norm_62, inplace=True)
        batch_norm_62 = None
        cat_6 = torch.cat([getitem_4, getitem_5, input_15, input_20], 1)
        getitem_4 = getitem_5 = input_15 = input_20 = None
        conv2d_63 = torch.conv2d(
            cat_6,
            l_self_modules_model_modules_6_modules_cv4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_6 = (
            l_self_modules_model_modules_6_modules_cv4_modules_conv_parameters_weight_
        ) = None
        batch_norm_63 = torch.nn.functional.batch_norm(
            conv2d_63,
            l_self_modules_model_modules_6_modules_cv4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_63 = (
            l_self_modules_model_modules_6_modules_cv4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_cv4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_cv4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_6_modules_cv4_modules_bn_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.silu(batch_norm_63, inplace=True)
        batch_norm_63 = None
        x_9 = torch._C._nn.avg_pool2d(x_8, 2, 1, 0, False, True)
        conv2d_64 = torch.conv2d(
            x_9,
            l_self_modules_model_modules_7_modules_cv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_9 = (
            l_self_modules_model_modules_7_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_64 = torch.nn.functional.batch_norm(
            conv2d_64,
            l_self_modules_model_modules_7_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_7_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_7_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_7_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_64 = (
            l_self_modules_model_modules_7_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_7_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_7_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_7_modules_cv1_modules_bn_parameters_bias_
        ) = None
        x_10 = torch.nn.functional.silu(batch_norm_64, inplace=True)
        batch_norm_64 = None
        conv2d_65 = torch.conv2d(
            x_10,
            l_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = (
            l_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_65 = torch.nn.functional.batch_norm(
            conv2d_65,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_65 = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_53 = torch.nn.functional.silu(batch_norm_65, inplace=True)
        batch_norm_65 = None
        chunk_3 = silu_53.chunk(2, 1)
        silu_53 = None
        getitem_6 = chunk_3[0]
        getitem_7 = chunk_3[1]
        chunk_3 = None
        conv2d_66 = torch.conv2d(
            getitem_7,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_66 = torch.nn.functional.batch_norm(
            conv2d_66,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_66 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_54 = torch.nn.functional.silu(batch_norm_66, inplace=True)
        batch_norm_66 = None
        conv2d_67 = torch.conv2d(
            silu_54,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_67 = torch.nn.functional.batch_norm(
            conv2d_67,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_67 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_68 = torch.conv2d(
            silu_54,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_68 = torch.nn.functional.batch_norm(
            conv2d_68,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_68 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_36 = batch_norm_67 + batch_norm_68
        batch_norm_67 = batch_norm_68 = None
        add_37 = add_36 + 0
        add_36 = None
        silu_55 = torch.nn.functional.silu(add_37, inplace=True)
        add_37 = None
        conv2d_69 = torch.conv2d(
            silu_55,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_55 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_69 = torch.nn.functional.batch_norm(
            conv2d_69,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_69 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_56 = torch.nn.functional.silu(batch_norm_69, inplace=True)
        batch_norm_69 = None
        input_21 = silu_54 + silu_56
        silu_54 = silu_56 = None
        conv2d_70 = torch.conv2d(
            input_21,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_70 = torch.nn.functional.batch_norm(
            conv2d_70,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_70 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_71 = torch.conv2d(
            input_21,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_71 = torch.nn.functional.batch_norm(
            conv2d_71,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_71 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_39 = batch_norm_70 + batch_norm_71
        batch_norm_70 = batch_norm_71 = None
        add_40 = add_39 + 0
        add_39 = None
        silu_57 = torch.nn.functional.silu(add_40, inplace=True)
        add_40 = None
        conv2d_72 = torch.conv2d(
            silu_57,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_57 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_72 = torch.nn.functional.batch_norm(
            conv2d_72,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_72 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_58 = torch.nn.functional.silu(batch_norm_72, inplace=True)
        batch_norm_72 = None
        input_22 = input_21 + silu_58
        input_21 = silu_58 = None
        conv2d_73 = torch.conv2d(
            input_22,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_73 = torch.nn.functional.batch_norm(
            conv2d_73,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_73 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_74 = torch.conv2d(
            input_22,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_74 = torch.nn.functional.batch_norm(
            conv2d_74,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_74 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_42 = batch_norm_73 + batch_norm_74
        batch_norm_73 = batch_norm_74 = None
        add_43 = add_42 + 0
        add_42 = None
        silu_59 = torch.nn.functional.silu(add_43, inplace=True)
        add_43 = None
        conv2d_75 = torch.conv2d(
            silu_59,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_59 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_75 = torch.nn.functional.batch_norm(
            conv2d_75,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_75 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_60 = torch.nn.functional.silu(batch_norm_75, inplace=True)
        batch_norm_75 = None
        input_23 = input_22 + silu_60
        input_22 = silu_60 = None
        conv2d_76 = torch.conv2d(
            getitem_7,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_76 = torch.nn.functional.batch_norm(
            conv2d_76,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_76 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_61 = torch.nn.functional.silu(batch_norm_76, inplace=True)
        batch_norm_76 = None
        cat_7 = torch.cat((input_23, silu_61), 1)
        input_23 = silu_61 = None
        conv2d_77 = torch.conv2d(
            cat_7,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_7 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_77 = torch.nn.functional.batch_norm(
            conv2d_77,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_77 = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_24 = torch.nn.functional.silu(batch_norm_77, inplace=True)
        batch_norm_77 = None
        conv2d_78 = torch.conv2d(
            input_24,
            l_self_modules_model_modules_8_modules_cv2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_24 = l_self_modules_model_modules_8_modules_cv2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_78 = torch.nn.functional.batch_norm(
            conv2d_78,
            l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_78 = l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv2_modules_1_modules_bn_parameters_bias_ = (None)
        input_25 = torch.nn.functional.silu(batch_norm_78, inplace=True)
        batch_norm_78 = None
        conv2d_79 = torch.conv2d(
            getitem_7,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_79 = torch.nn.functional.batch_norm(
            conv2d_79,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_79 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_64 = torch.nn.functional.silu(batch_norm_79, inplace=True)
        batch_norm_79 = None
        conv2d_80 = torch.conv2d(
            silu_64,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_80 = torch.nn.functional.batch_norm(
            conv2d_80,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_80 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_81 = torch.conv2d(
            silu_64,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_81 = torch.nn.functional.batch_norm(
            conv2d_81,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_81 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_45 = batch_norm_80 + batch_norm_81
        batch_norm_80 = batch_norm_81 = None
        add_46 = add_45 + 0
        add_45 = None
        silu_65 = torch.nn.functional.silu(add_46, inplace=True)
        add_46 = None
        conv2d_82 = torch.conv2d(
            silu_65,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_65 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_82 = torch.nn.functional.batch_norm(
            conv2d_82,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_82 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_66 = torch.nn.functional.silu(batch_norm_82, inplace=True)
        batch_norm_82 = None
        input_26 = silu_64 + silu_66
        silu_64 = silu_66 = None
        conv2d_83 = torch.conv2d(
            input_26,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_83 = torch.nn.functional.batch_norm(
            conv2d_83,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_83 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_84 = torch.conv2d(
            input_26,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_84 = torch.nn.functional.batch_norm(
            conv2d_84,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_84 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_48 = batch_norm_83 + batch_norm_84
        batch_norm_83 = batch_norm_84 = None
        add_49 = add_48 + 0
        add_48 = None
        silu_67 = torch.nn.functional.silu(add_49, inplace=True)
        add_49 = None
        conv2d_85 = torch.conv2d(
            silu_67,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_67 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_85 = torch.nn.functional.batch_norm(
            conv2d_85,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_85 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_68 = torch.nn.functional.silu(batch_norm_85, inplace=True)
        batch_norm_85 = None
        input_27 = input_26 + silu_68
        input_26 = silu_68 = None
        conv2d_86 = torch.conv2d(
            input_27,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_86 = torch.nn.functional.batch_norm(
            conv2d_86,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_86 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_87 = torch.conv2d(
            input_27,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_87 = torch.nn.functional.batch_norm(
            conv2d_87,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_87 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_51 = batch_norm_86 + batch_norm_87
        batch_norm_86 = batch_norm_87 = None
        add_52 = add_51 + 0
        add_51 = None
        silu_69 = torch.nn.functional.silu(add_52, inplace=True)
        add_52 = None
        conv2d_88 = torch.conv2d(
            silu_69,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_69 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_88 = torch.nn.functional.batch_norm(
            conv2d_88,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_88 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_70 = torch.nn.functional.silu(batch_norm_88, inplace=True)
        batch_norm_88 = None
        input_28 = input_27 + silu_70
        input_27 = silu_70 = None
        conv2d_89 = torch.conv2d(
            getitem_7,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_89 = torch.nn.functional.batch_norm(
            conv2d_89,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_89 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_71 = torch.nn.functional.silu(batch_norm_89, inplace=True)
        batch_norm_89 = None
        cat_8 = torch.cat((input_28, silu_71), 1)
        input_28 = silu_71 = None
        conv2d_90 = torch.conv2d(
            cat_8,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_8 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_90 = torch.nn.functional.batch_norm(
            conv2d_90,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_90 = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_29 = torch.nn.functional.silu(batch_norm_90, inplace=True)
        batch_norm_90 = None
        conv2d_91 = torch.conv2d(
            input_29,
            l_self_modules_model_modules_8_modules_cv3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_29 = l_self_modules_model_modules_8_modules_cv3_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_91 = torch.nn.functional.batch_norm(
            conv2d_91,
            l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_91 = l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_cv3_modules_1_modules_bn_parameters_bias_ = (None)
        input_30 = torch.nn.functional.silu(batch_norm_91, inplace=True)
        batch_norm_91 = None
        cat_9 = torch.cat([getitem_6, getitem_7, input_25, input_30], 1)
        getitem_6 = getitem_7 = input_25 = input_30 = None
        conv2d_92 = torch.conv2d(
            cat_9,
            l_self_modules_model_modules_8_modules_cv4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_9 = (
            l_self_modules_model_modules_8_modules_cv4_modules_conv_parameters_weight_
        ) = None
        batch_norm_92 = torch.nn.functional.batch_norm(
            conv2d_92,
            l_self_modules_model_modules_8_modules_cv4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_92 = (
            l_self_modules_model_modules_8_modules_cv4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv4_modules_bn_parameters_bias_
        ) = None
        x_11 = torch.nn.functional.silu(batch_norm_92, inplace=True)
        batch_norm_92 = None
        conv2d_93 = torch.conv2d(
            x_11,
            l_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = (
            l_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_93 = torch.nn.functional.batch_norm(
            conv2d_93,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_93 = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_75 = torch.nn.functional.silu(batch_norm_93, inplace=True)
        batch_norm_93 = None
        max_pool2d = torch.nn.functional.max_pool2d(
            silu_75, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_1 = torch.nn.functional.max_pool2d(
            silu_75, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_2 = torch.nn.functional.max_pool2d(
            silu_75, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        cat_10 = torch.cat([silu_75, max_pool2d, max_pool2d_1, max_pool2d_2], 1)
        silu_75 = max_pool2d = max_pool2d_1 = max_pool2d_2 = None
        conv2d_94 = torch.conv2d(
            cat_10,
            l_self_modules_model_modules_9_modules_cv5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_10 = (
            l_self_modules_model_modules_9_modules_cv5_modules_conv_parameters_weight_
        ) = None
        batch_norm_94 = torch.nn.functional.batch_norm(
            conv2d_94,
            l_self_modules_model_modules_9_modules_cv5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_cv5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_cv5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_cv5_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_94 = (
            l_self_modules_model_modules_9_modules_cv5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_cv5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_cv5_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_cv5_modules_bn_parameters_bias_
        ) = None
        x_12 = torch.nn.functional.silu(batch_norm_94, inplace=True)
        batch_norm_94 = None
        x_13 = torch.nn.functional.interpolate(
            x_12, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        x_14 = torch.cat([x_13, x_8], 1)
        x_13 = x_8 = None
        conv2d_95 = torch.conv2d(
            x_14,
            l_self_modules_model_modules_12_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = (
            l_self_modules_model_modules_12_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_95 = torch.nn.functional.batch_norm(
            conv2d_95,
            l_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_95 = (
            l_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_77 = torch.nn.functional.silu(batch_norm_95, inplace=True)
        batch_norm_95 = None
        chunk_4 = silu_77.chunk(2, 1)
        silu_77 = None
        getitem_8 = chunk_4[0]
        getitem_9 = chunk_4[1]
        chunk_4 = None
        conv2d_96 = torch.conv2d(
            getitem_9,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_96 = torch.nn.functional.batch_norm(
            conv2d_96,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_96 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_78 = torch.nn.functional.silu(batch_norm_96, inplace=True)
        batch_norm_96 = None
        conv2d_97 = torch.conv2d(
            silu_78,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_97 = torch.nn.functional.batch_norm(
            conv2d_97,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_97 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_98 = torch.conv2d(
            silu_78,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_98 = torch.nn.functional.batch_norm(
            conv2d_98,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_98 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_54 = batch_norm_97 + batch_norm_98
        batch_norm_97 = batch_norm_98 = None
        add_55 = add_54 + 0
        add_54 = None
        silu_79 = torch.nn.functional.silu(add_55, inplace=True)
        add_55 = None
        conv2d_99 = torch.conv2d(
            silu_79,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_79 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_99 = torch.nn.functional.batch_norm(
            conv2d_99,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_99 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_80 = torch.nn.functional.silu(batch_norm_99, inplace=True)
        batch_norm_99 = None
        input_31 = silu_78 + silu_80
        silu_78 = silu_80 = None
        conv2d_100 = torch.conv2d(
            input_31,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_100 = torch.nn.functional.batch_norm(
            conv2d_100,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_100 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_101 = torch.conv2d(
            input_31,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_101 = torch.nn.functional.batch_norm(
            conv2d_101,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_101 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_57 = batch_norm_100 + batch_norm_101
        batch_norm_100 = batch_norm_101 = None
        add_58 = add_57 + 0
        add_57 = None
        silu_81 = torch.nn.functional.silu(add_58, inplace=True)
        add_58 = None
        conv2d_102 = torch.conv2d(
            silu_81,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_81 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_102 = torch.nn.functional.batch_norm(
            conv2d_102,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_102 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_82 = torch.nn.functional.silu(batch_norm_102, inplace=True)
        batch_norm_102 = None
        input_32 = input_31 + silu_82
        input_31 = silu_82 = None
        conv2d_103 = torch.conv2d(
            input_32,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_103 = torch.nn.functional.batch_norm(
            conv2d_103,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_103 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_104 = torch.conv2d(
            input_32,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_104 = torch.nn.functional.batch_norm(
            conv2d_104,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_104 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_60 = batch_norm_103 + batch_norm_104
        batch_norm_103 = batch_norm_104 = None
        add_61 = add_60 + 0
        add_60 = None
        silu_83 = torch.nn.functional.silu(add_61, inplace=True)
        add_61 = None
        conv2d_105 = torch.conv2d(
            silu_83,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_83 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_105 = torch.nn.functional.batch_norm(
            conv2d_105,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_105 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_84 = torch.nn.functional.silu(batch_norm_105, inplace=True)
        batch_norm_105 = None
        input_33 = input_32 + silu_84
        input_32 = silu_84 = None
        conv2d_106 = torch.conv2d(
            getitem_9,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_106 = torch.nn.functional.batch_norm(
            conv2d_106,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_106 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_85 = torch.nn.functional.silu(batch_norm_106, inplace=True)
        batch_norm_106 = None
        cat_12 = torch.cat((input_33, silu_85), 1)
        input_33 = silu_85 = None
        conv2d_107 = torch.conv2d(
            cat_12,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_12 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_107 = torch.nn.functional.batch_norm(
            conv2d_107,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_107 = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_34 = torch.nn.functional.silu(batch_norm_107, inplace=True)
        batch_norm_107 = None
        conv2d_108 = torch.conv2d(
            input_34,
            l_self_modules_model_modules_12_modules_cv2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_34 = l_self_modules_model_modules_12_modules_cv2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_108 = torch.nn.functional.batch_norm(
            conv2d_108,
            l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_108 = l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv2_modules_1_modules_bn_parameters_bias_ = (None)
        input_35 = torch.nn.functional.silu(batch_norm_108, inplace=True)
        batch_norm_108 = None
        conv2d_109 = torch.conv2d(
            getitem_9,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_109 = torch.nn.functional.batch_norm(
            conv2d_109,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_109 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_88 = torch.nn.functional.silu(batch_norm_109, inplace=True)
        batch_norm_109 = None
        conv2d_110 = torch.conv2d(
            silu_88,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_110 = torch.nn.functional.batch_norm(
            conv2d_110,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_110 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_111 = torch.conv2d(
            silu_88,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_111 = torch.nn.functional.batch_norm(
            conv2d_111,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_111 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_63 = batch_norm_110 + batch_norm_111
        batch_norm_110 = batch_norm_111 = None
        add_64 = add_63 + 0
        add_63 = None
        silu_89 = torch.nn.functional.silu(add_64, inplace=True)
        add_64 = None
        conv2d_112 = torch.conv2d(
            silu_89,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_89 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_112 = torch.nn.functional.batch_norm(
            conv2d_112,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_112 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_90 = torch.nn.functional.silu(batch_norm_112, inplace=True)
        batch_norm_112 = None
        input_36 = silu_88 + silu_90
        silu_88 = silu_90 = None
        conv2d_113 = torch.conv2d(
            input_36,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_113 = torch.nn.functional.batch_norm(
            conv2d_113,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_113 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_114 = torch.conv2d(
            input_36,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_114 = torch.nn.functional.batch_norm(
            conv2d_114,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_114 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_66 = batch_norm_113 + batch_norm_114
        batch_norm_113 = batch_norm_114 = None
        add_67 = add_66 + 0
        add_66 = None
        silu_91 = torch.nn.functional.silu(add_67, inplace=True)
        add_67 = None
        conv2d_115 = torch.conv2d(
            silu_91,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_91 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_115 = torch.nn.functional.batch_norm(
            conv2d_115,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_115 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_92 = torch.nn.functional.silu(batch_norm_115, inplace=True)
        batch_norm_115 = None
        input_37 = input_36 + silu_92
        input_36 = silu_92 = None
        conv2d_116 = torch.conv2d(
            input_37,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_116 = torch.nn.functional.batch_norm(
            conv2d_116,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_116 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_117 = torch.conv2d(
            input_37,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_117 = torch.nn.functional.batch_norm(
            conv2d_117,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_117 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_69 = batch_norm_116 + batch_norm_117
        batch_norm_116 = batch_norm_117 = None
        add_70 = add_69 + 0
        add_69 = None
        silu_93 = torch.nn.functional.silu(add_70, inplace=True)
        add_70 = None
        conv2d_118 = torch.conv2d(
            silu_93,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_93 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_118 = torch.nn.functional.batch_norm(
            conv2d_118,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_118 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_94 = torch.nn.functional.silu(batch_norm_118, inplace=True)
        batch_norm_118 = None
        input_38 = input_37 + silu_94
        input_37 = silu_94 = None
        conv2d_119 = torch.conv2d(
            getitem_9,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_119 = torch.nn.functional.batch_norm(
            conv2d_119,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_119 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_95 = torch.nn.functional.silu(batch_norm_119, inplace=True)
        batch_norm_119 = None
        cat_13 = torch.cat((input_38, silu_95), 1)
        input_38 = silu_95 = None
        conv2d_120 = torch.conv2d(
            cat_13,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_13 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_120 = torch.nn.functional.batch_norm(
            conv2d_120,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_120 = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_39 = torch.nn.functional.silu(batch_norm_120, inplace=True)
        batch_norm_120 = None
        conv2d_121 = torch.conv2d(
            input_39,
            l_self_modules_model_modules_12_modules_cv3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_39 = l_self_modules_model_modules_12_modules_cv3_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_121 = torch.nn.functional.batch_norm(
            conv2d_121,
            l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_121 = l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_12_modules_cv3_modules_1_modules_bn_parameters_bias_ = (None)
        input_40 = torch.nn.functional.silu(batch_norm_121, inplace=True)
        batch_norm_121 = None
        cat_14 = torch.cat([getitem_8, getitem_9, input_35, input_40], 1)
        getitem_8 = getitem_9 = input_35 = input_40 = None
        conv2d_122 = torch.conv2d(
            cat_14,
            l_self_modules_model_modules_12_modules_cv4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_14 = (
            l_self_modules_model_modules_12_modules_cv4_modules_conv_parameters_weight_
        ) = None
        batch_norm_122 = torch.nn.functional.batch_norm(
            conv2d_122,
            l_self_modules_model_modules_12_modules_cv4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_12_modules_cv4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_12_modules_cv4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_12_modules_cv4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_122 = (
            l_self_modules_model_modules_12_modules_cv4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_12_modules_cv4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_12_modules_cv4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_12_modules_cv4_modules_bn_parameters_bias_
        ) = None
        x_15 = torch.nn.functional.silu(batch_norm_122, inplace=True)
        batch_norm_122 = None
        x_16 = torch.nn.functional.interpolate(
            x_15, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        x_17 = torch.cat([x_16, x_5], 1)
        x_16 = x_5 = None
        conv2d_123 = torch.conv2d(
            x_17,
            l_self_modules_model_modules_15_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = (
            l_self_modules_model_modules_15_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_123 = torch.nn.functional.batch_norm(
            conv2d_123,
            l_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_123 = (
            l_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_99 = torch.nn.functional.silu(batch_norm_123, inplace=True)
        batch_norm_123 = None
        chunk_5 = silu_99.chunk(2, 1)
        silu_99 = None
        getitem_10 = chunk_5[0]
        getitem_11 = chunk_5[1]
        chunk_5 = None
        conv2d_124 = torch.conv2d(
            getitem_11,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_124 = torch.nn.functional.batch_norm(
            conv2d_124,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_124 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_100 = torch.nn.functional.silu(batch_norm_124, inplace=True)
        batch_norm_124 = None
        conv2d_125 = torch.conv2d(
            silu_100,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_125 = torch.nn.functional.batch_norm(
            conv2d_125,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_125 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_126 = torch.conv2d(
            silu_100,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_126 = torch.nn.functional.batch_norm(
            conv2d_126,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_126 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_72 = batch_norm_125 + batch_norm_126
        batch_norm_125 = batch_norm_126 = None
        add_73 = add_72 + 0
        add_72 = None
        silu_101 = torch.nn.functional.silu(add_73, inplace=True)
        add_73 = None
        conv2d_127 = torch.conv2d(
            silu_101,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_101 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_127 = torch.nn.functional.batch_norm(
            conv2d_127,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_127 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_102 = torch.nn.functional.silu(batch_norm_127, inplace=True)
        batch_norm_127 = None
        input_41 = silu_100 + silu_102
        silu_100 = silu_102 = None
        conv2d_128 = torch.conv2d(
            input_41,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_128 = torch.nn.functional.batch_norm(
            conv2d_128,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_128 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_129 = torch.conv2d(
            input_41,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_129 = torch.nn.functional.batch_norm(
            conv2d_129,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_129 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_75 = batch_norm_128 + batch_norm_129
        batch_norm_128 = batch_norm_129 = None
        add_76 = add_75 + 0
        add_75 = None
        silu_103 = torch.nn.functional.silu(add_76, inplace=True)
        add_76 = None
        conv2d_130 = torch.conv2d(
            silu_103,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_103 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_130 = torch.nn.functional.batch_norm(
            conv2d_130,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_130 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_104 = torch.nn.functional.silu(batch_norm_130, inplace=True)
        batch_norm_130 = None
        input_42 = input_41 + silu_104
        input_41 = silu_104 = None
        conv2d_131 = torch.conv2d(
            input_42,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_131 = torch.nn.functional.batch_norm(
            conv2d_131,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_131 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_132 = torch.conv2d(
            input_42,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_132 = torch.nn.functional.batch_norm(
            conv2d_132,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_132 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_78 = batch_norm_131 + batch_norm_132
        batch_norm_131 = batch_norm_132 = None
        add_79 = add_78 + 0
        add_78 = None
        silu_105 = torch.nn.functional.silu(add_79, inplace=True)
        add_79 = None
        conv2d_133 = torch.conv2d(
            silu_105,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_105 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_133 = torch.nn.functional.batch_norm(
            conv2d_133,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_133 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_106 = torch.nn.functional.silu(batch_norm_133, inplace=True)
        batch_norm_133 = None
        input_43 = input_42 + silu_106
        input_42 = silu_106 = None
        conv2d_134 = torch.conv2d(
            getitem_11,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_134 = torch.nn.functional.batch_norm(
            conv2d_134,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_134 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_107 = torch.nn.functional.silu(batch_norm_134, inplace=True)
        batch_norm_134 = None
        cat_16 = torch.cat((input_43, silu_107), 1)
        input_43 = silu_107 = None
        conv2d_135 = torch.conv2d(
            cat_16,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_16 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_135 = torch.nn.functional.batch_norm(
            conv2d_135,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_135 = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_44 = torch.nn.functional.silu(batch_norm_135, inplace=True)
        batch_norm_135 = None
        conv2d_136 = torch.conv2d(
            input_44,
            l_self_modules_model_modules_15_modules_cv2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_44 = l_self_modules_model_modules_15_modules_cv2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_136 = torch.nn.functional.batch_norm(
            conv2d_136,
            l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_136 = l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv2_modules_1_modules_bn_parameters_bias_ = (None)
        input_45 = torch.nn.functional.silu(batch_norm_136, inplace=True)
        batch_norm_136 = None
        conv2d_137 = torch.conv2d(
            getitem_11,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_137 = torch.nn.functional.batch_norm(
            conv2d_137,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_137 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_110 = torch.nn.functional.silu(batch_norm_137, inplace=True)
        batch_norm_137 = None
        conv2d_138 = torch.conv2d(
            silu_110,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_138 = torch.nn.functional.batch_norm(
            conv2d_138,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_138 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_139 = torch.conv2d(
            silu_110,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_139 = torch.nn.functional.batch_norm(
            conv2d_139,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_139 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_81 = batch_norm_138 + batch_norm_139
        batch_norm_138 = batch_norm_139 = None
        add_82 = add_81 + 0
        add_81 = None
        silu_111 = torch.nn.functional.silu(add_82, inplace=True)
        add_82 = None
        conv2d_140 = torch.conv2d(
            silu_111,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_111 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_140 = torch.nn.functional.batch_norm(
            conv2d_140,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_140 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_112 = torch.nn.functional.silu(batch_norm_140, inplace=True)
        batch_norm_140 = None
        input_46 = silu_110 + silu_112
        silu_110 = silu_112 = None
        conv2d_141 = torch.conv2d(
            input_46,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_141 = torch.nn.functional.batch_norm(
            conv2d_141,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_141 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_142 = torch.conv2d(
            input_46,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_142 = torch.nn.functional.batch_norm(
            conv2d_142,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_142 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_84 = batch_norm_141 + batch_norm_142
        batch_norm_141 = batch_norm_142 = None
        add_85 = add_84 + 0
        add_84 = None
        silu_113 = torch.nn.functional.silu(add_85, inplace=True)
        add_85 = None
        conv2d_143 = torch.conv2d(
            silu_113,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_113 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_143 = torch.nn.functional.batch_norm(
            conv2d_143,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_143 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_114 = torch.nn.functional.silu(batch_norm_143, inplace=True)
        batch_norm_143 = None
        input_47 = input_46 + silu_114
        input_46 = silu_114 = None
        conv2d_144 = torch.conv2d(
            input_47,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_144 = torch.nn.functional.batch_norm(
            conv2d_144,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_144 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_145 = torch.conv2d(
            input_47,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_145 = torch.nn.functional.batch_norm(
            conv2d_145,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_145 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_87 = batch_norm_144 + batch_norm_145
        batch_norm_144 = batch_norm_145 = None
        add_88 = add_87 + 0
        add_87 = None
        silu_115 = torch.nn.functional.silu(add_88, inplace=True)
        add_88 = None
        conv2d_146 = torch.conv2d(
            silu_115,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_115 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_146 = torch.nn.functional.batch_norm(
            conv2d_146,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_146 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_116 = torch.nn.functional.silu(batch_norm_146, inplace=True)
        batch_norm_146 = None
        input_48 = input_47 + silu_116
        input_47 = silu_116 = None
        conv2d_147 = torch.conv2d(
            getitem_11,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_147 = torch.nn.functional.batch_norm(
            conv2d_147,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_147 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_117 = torch.nn.functional.silu(batch_norm_147, inplace=True)
        batch_norm_147 = None
        cat_17 = torch.cat((input_48, silu_117), 1)
        input_48 = silu_117 = None
        conv2d_148 = torch.conv2d(
            cat_17,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_17 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_148 = torch.nn.functional.batch_norm(
            conv2d_148,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_148 = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_49 = torch.nn.functional.silu(batch_norm_148, inplace=True)
        batch_norm_148 = None
        conv2d_149 = torch.conv2d(
            input_49,
            l_self_modules_model_modules_15_modules_cv3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_49 = l_self_modules_model_modules_15_modules_cv3_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_149 = torch.nn.functional.batch_norm(
            conv2d_149,
            l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_149 = l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_15_modules_cv3_modules_1_modules_bn_parameters_bias_ = (None)
        input_50 = torch.nn.functional.silu(batch_norm_149, inplace=True)
        batch_norm_149 = None
        cat_18 = torch.cat([getitem_10, getitem_11, input_45, input_50], 1)
        getitem_10 = getitem_11 = input_45 = input_50 = None
        conv2d_150 = torch.conv2d(
            cat_18,
            l_self_modules_model_modules_15_modules_cv4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_18 = (
            l_self_modules_model_modules_15_modules_cv4_modules_conv_parameters_weight_
        ) = None
        batch_norm_150 = torch.nn.functional.batch_norm(
            conv2d_150,
            l_self_modules_model_modules_15_modules_cv4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_cv4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_cv4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_cv4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_150 = (
            l_self_modules_model_modules_15_modules_cv4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_15_modules_cv4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_15_modules_cv4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_15_modules_cv4_modules_bn_parameters_bias_
        ) = None
        x_18 = torch.nn.functional.silu(batch_norm_150, inplace=True)
        batch_norm_150 = None
        x_19 = torch._C._nn.avg_pool2d(x_18, 2, 1, 0, False, True)
        conv2d_151 = torch.conv2d(
            x_19,
            l_self_modules_model_modules_16_modules_cv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_19 = (
            l_self_modules_model_modules_16_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_151 = torch.nn.functional.batch_norm(
            conv2d_151,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_151 = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_bias_
        ) = None
        x_20 = torch.nn.functional.silu(batch_norm_151, inplace=True)
        batch_norm_151 = None
        x_21 = torch.cat([x_20, x_15], 1)
        x_20 = x_15 = None
        conv2d_152 = torch.conv2d(
            x_21,
            l_self_modules_model_modules_18_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = (
            l_self_modules_model_modules_18_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_152 = torch.nn.functional.batch_norm(
            conv2d_152,
            l_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_152 = (
            l_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_122 = torch.nn.functional.silu(batch_norm_152, inplace=True)
        batch_norm_152 = None
        chunk_6 = silu_122.chunk(2, 1)
        silu_122 = None
        getitem_12 = chunk_6[0]
        getitem_13 = chunk_6[1]
        chunk_6 = None
        conv2d_153 = torch.conv2d(
            getitem_13,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_153 = torch.nn.functional.batch_norm(
            conv2d_153,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_153 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_123 = torch.nn.functional.silu(batch_norm_153, inplace=True)
        batch_norm_153 = None
        conv2d_154 = torch.conv2d(
            silu_123,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_154 = torch.nn.functional.batch_norm(
            conv2d_154,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_154 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_155 = torch.conv2d(
            silu_123,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_155 = torch.nn.functional.batch_norm(
            conv2d_155,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_155 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_90 = batch_norm_154 + batch_norm_155
        batch_norm_154 = batch_norm_155 = None
        add_91 = add_90 + 0
        add_90 = None
        silu_124 = torch.nn.functional.silu(add_91, inplace=True)
        add_91 = None
        conv2d_156 = torch.conv2d(
            silu_124,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_124 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_156 = torch.nn.functional.batch_norm(
            conv2d_156,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_156 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_125 = torch.nn.functional.silu(batch_norm_156, inplace=True)
        batch_norm_156 = None
        input_51 = silu_123 + silu_125
        silu_123 = silu_125 = None
        conv2d_157 = torch.conv2d(
            input_51,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_157 = torch.nn.functional.batch_norm(
            conv2d_157,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_157 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_158 = torch.conv2d(
            input_51,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_158 = torch.nn.functional.batch_norm(
            conv2d_158,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_158 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_93 = batch_norm_157 + batch_norm_158
        batch_norm_157 = batch_norm_158 = None
        add_94 = add_93 + 0
        add_93 = None
        silu_126 = torch.nn.functional.silu(add_94, inplace=True)
        add_94 = None
        conv2d_159 = torch.conv2d(
            silu_126,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_126 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_159 = torch.nn.functional.batch_norm(
            conv2d_159,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_159 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_127 = torch.nn.functional.silu(batch_norm_159, inplace=True)
        batch_norm_159 = None
        input_52 = input_51 + silu_127
        input_51 = silu_127 = None
        conv2d_160 = torch.conv2d(
            input_52,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_160 = torch.nn.functional.batch_norm(
            conv2d_160,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_160 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_161 = torch.conv2d(
            input_52,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_161 = torch.nn.functional.batch_norm(
            conv2d_161,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_161 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_96 = batch_norm_160 + batch_norm_161
        batch_norm_160 = batch_norm_161 = None
        add_97 = add_96 + 0
        add_96 = None
        silu_128 = torch.nn.functional.silu(add_97, inplace=True)
        add_97 = None
        conv2d_162 = torch.conv2d(
            silu_128,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_128 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_162 = torch.nn.functional.batch_norm(
            conv2d_162,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_162 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_129 = torch.nn.functional.silu(batch_norm_162, inplace=True)
        batch_norm_162 = None
        input_53 = input_52 + silu_129
        input_52 = silu_129 = None
        conv2d_163 = torch.conv2d(
            getitem_13,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_163 = torch.nn.functional.batch_norm(
            conv2d_163,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_163 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_130 = torch.nn.functional.silu(batch_norm_163, inplace=True)
        batch_norm_163 = None
        cat_20 = torch.cat((input_53, silu_130), 1)
        input_53 = silu_130 = None
        conv2d_164 = torch.conv2d(
            cat_20,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_20 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_164 = torch.nn.functional.batch_norm(
            conv2d_164,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_164 = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_54 = torch.nn.functional.silu(batch_norm_164, inplace=True)
        batch_norm_164 = None
        conv2d_165 = torch.conv2d(
            input_54,
            l_self_modules_model_modules_18_modules_cv2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_54 = l_self_modules_model_modules_18_modules_cv2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_165 = torch.nn.functional.batch_norm(
            conv2d_165,
            l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_165 = l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv2_modules_1_modules_bn_parameters_bias_ = (None)
        input_55 = torch.nn.functional.silu(batch_norm_165, inplace=True)
        batch_norm_165 = None
        conv2d_166 = torch.conv2d(
            getitem_13,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_166 = torch.nn.functional.batch_norm(
            conv2d_166,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_166 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_133 = torch.nn.functional.silu(batch_norm_166, inplace=True)
        batch_norm_166 = None
        conv2d_167 = torch.conv2d(
            silu_133,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_167 = torch.nn.functional.batch_norm(
            conv2d_167,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_167 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_168 = torch.conv2d(
            silu_133,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_168 = torch.nn.functional.batch_norm(
            conv2d_168,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_168 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_99 = batch_norm_167 + batch_norm_168
        batch_norm_167 = batch_norm_168 = None
        add_100 = add_99 + 0
        add_99 = None
        silu_134 = torch.nn.functional.silu(add_100, inplace=True)
        add_100 = None
        conv2d_169 = torch.conv2d(
            silu_134,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_134 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_169 = torch.nn.functional.batch_norm(
            conv2d_169,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_169 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_135 = torch.nn.functional.silu(batch_norm_169, inplace=True)
        batch_norm_169 = None
        input_56 = silu_133 + silu_135
        silu_133 = silu_135 = None
        conv2d_170 = torch.conv2d(
            input_56,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_170 = torch.nn.functional.batch_norm(
            conv2d_170,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_170 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_171 = torch.conv2d(
            input_56,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_171 = torch.nn.functional.batch_norm(
            conv2d_171,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_171 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_102 = batch_norm_170 + batch_norm_171
        batch_norm_170 = batch_norm_171 = None
        add_103 = add_102 + 0
        add_102 = None
        silu_136 = torch.nn.functional.silu(add_103, inplace=True)
        add_103 = None
        conv2d_172 = torch.conv2d(
            silu_136,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_136 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_172 = torch.nn.functional.batch_norm(
            conv2d_172,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_172 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_137 = torch.nn.functional.silu(batch_norm_172, inplace=True)
        batch_norm_172 = None
        input_57 = input_56 + silu_137
        input_56 = silu_137 = None
        conv2d_173 = torch.conv2d(
            input_57,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_173 = torch.nn.functional.batch_norm(
            conv2d_173,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_173 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_174 = torch.conv2d(
            input_57,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_174 = torch.nn.functional.batch_norm(
            conv2d_174,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_174 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_105 = batch_norm_173 + batch_norm_174
        batch_norm_173 = batch_norm_174 = None
        add_106 = add_105 + 0
        add_105 = None
        silu_138 = torch.nn.functional.silu(add_106, inplace=True)
        add_106 = None
        conv2d_175 = torch.conv2d(
            silu_138,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_138 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_175 = torch.nn.functional.batch_norm(
            conv2d_175,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_175 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_139 = torch.nn.functional.silu(batch_norm_175, inplace=True)
        batch_norm_175 = None
        input_58 = input_57 + silu_139
        input_57 = silu_139 = None
        conv2d_176 = torch.conv2d(
            getitem_13,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_176 = torch.nn.functional.batch_norm(
            conv2d_176,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_176 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_140 = torch.nn.functional.silu(batch_norm_176, inplace=True)
        batch_norm_176 = None
        cat_21 = torch.cat((input_58, silu_140), 1)
        input_58 = silu_140 = None
        conv2d_177 = torch.conv2d(
            cat_21,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_21 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_177 = torch.nn.functional.batch_norm(
            conv2d_177,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_177 = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_59 = torch.nn.functional.silu(batch_norm_177, inplace=True)
        batch_norm_177 = None
        conv2d_178 = torch.conv2d(
            input_59,
            l_self_modules_model_modules_18_modules_cv3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_59 = l_self_modules_model_modules_18_modules_cv3_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_178 = torch.nn.functional.batch_norm(
            conv2d_178,
            l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_178 = l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_18_modules_cv3_modules_1_modules_bn_parameters_bias_ = (None)
        input_60 = torch.nn.functional.silu(batch_norm_178, inplace=True)
        batch_norm_178 = None
        cat_22 = torch.cat([getitem_12, getitem_13, input_55, input_60], 1)
        getitem_12 = getitem_13 = input_55 = input_60 = None
        conv2d_179 = torch.conv2d(
            cat_22,
            l_self_modules_model_modules_18_modules_cv4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_22 = (
            l_self_modules_model_modules_18_modules_cv4_modules_conv_parameters_weight_
        ) = None
        batch_norm_179 = torch.nn.functional.batch_norm(
            conv2d_179,
            l_self_modules_model_modules_18_modules_cv4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_cv4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_cv4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_cv4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_179 = (
            l_self_modules_model_modules_18_modules_cv4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_18_modules_cv4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_18_modules_cv4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_18_modules_cv4_modules_bn_parameters_bias_
        ) = None
        x_22 = torch.nn.functional.silu(batch_norm_179, inplace=True)
        batch_norm_179 = None
        x_23 = torch._C._nn.avg_pool2d(x_22, 2, 1, 0, False, True)
        conv2d_180 = torch.conv2d(
            x_23,
            l_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_23 = (
            l_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_180 = torch.nn.functional.batch_norm(
            conv2d_180,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_180 = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_
        ) = None
        x_24 = torch.nn.functional.silu(batch_norm_180, inplace=True)
        batch_norm_180 = None
        x_25 = torch.cat([x_24, x_12], 1)
        x_24 = x_12 = None
        conv2d_181 = torch.conv2d(
            x_25,
            l_self_modules_model_modules_21_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = (
            l_self_modules_model_modules_21_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_181 = torch.nn.functional.batch_norm(
            conv2d_181,
            l_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_181 = (
            l_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_145 = torch.nn.functional.silu(batch_norm_181, inplace=True)
        batch_norm_181 = None
        chunk_7 = silu_145.chunk(2, 1)
        silu_145 = None
        getitem_14 = chunk_7[0]
        getitem_15 = chunk_7[1]
        chunk_7 = None
        conv2d_182 = torch.conv2d(
            getitem_15,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_182 = torch.nn.functional.batch_norm(
            conv2d_182,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_182 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_146 = torch.nn.functional.silu(batch_norm_182, inplace=True)
        batch_norm_182 = None
        conv2d_183 = torch.conv2d(
            silu_146,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_183 = torch.nn.functional.batch_norm(
            conv2d_183,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_183 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_184 = torch.conv2d(
            silu_146,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_184 = torch.nn.functional.batch_norm(
            conv2d_184,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_184 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_108 = batch_norm_183 + batch_norm_184
        batch_norm_183 = batch_norm_184 = None
        add_109 = add_108 + 0
        add_108 = None
        silu_147 = torch.nn.functional.silu(add_109, inplace=True)
        add_109 = None
        conv2d_185 = torch.conv2d(
            silu_147,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_147 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_185 = torch.nn.functional.batch_norm(
            conv2d_185,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_185 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_148 = torch.nn.functional.silu(batch_norm_185, inplace=True)
        batch_norm_185 = None
        input_61 = silu_146 + silu_148
        silu_146 = silu_148 = None
        conv2d_186 = torch.conv2d(
            input_61,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_186 = torch.nn.functional.batch_norm(
            conv2d_186,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_186 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_187 = torch.conv2d(
            input_61,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_187 = torch.nn.functional.batch_norm(
            conv2d_187,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_187 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_111 = batch_norm_186 + batch_norm_187
        batch_norm_186 = batch_norm_187 = None
        add_112 = add_111 + 0
        add_111 = None
        silu_149 = torch.nn.functional.silu(add_112, inplace=True)
        add_112 = None
        conv2d_188 = torch.conv2d(
            silu_149,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_149 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_188 = torch.nn.functional.batch_norm(
            conv2d_188,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_188 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_150 = torch.nn.functional.silu(batch_norm_188, inplace=True)
        batch_norm_188 = None
        input_62 = input_61 + silu_150
        input_61 = silu_150 = None
        conv2d_189 = torch.conv2d(
            input_62,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_189 = torch.nn.functional.batch_norm(
            conv2d_189,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_189 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_190 = torch.conv2d(
            input_62,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_190 = torch.nn.functional.batch_norm(
            conv2d_190,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_190 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_114 = batch_norm_189 + batch_norm_190
        batch_norm_189 = batch_norm_190 = None
        add_115 = add_114 + 0
        add_114 = None
        silu_151 = torch.nn.functional.silu(add_115, inplace=True)
        add_115 = None
        conv2d_191 = torch.conv2d(
            silu_151,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_151 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_191 = torch.nn.functional.batch_norm(
            conv2d_191,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_191 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_152 = torch.nn.functional.silu(batch_norm_191, inplace=True)
        batch_norm_191 = None
        input_63 = input_62 + silu_152
        input_62 = silu_152 = None
        conv2d_192 = torch.conv2d(
            getitem_15,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_192 = torch.nn.functional.batch_norm(
            conv2d_192,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_192 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_153 = torch.nn.functional.silu(batch_norm_192, inplace=True)
        batch_norm_192 = None
        cat_24 = torch.cat((input_63, silu_153), 1)
        input_63 = silu_153 = None
        conv2d_193 = torch.conv2d(
            cat_24,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_24 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_193 = torch.nn.functional.batch_norm(
            conv2d_193,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_193 = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_64 = torch.nn.functional.silu(batch_norm_193, inplace=True)
        batch_norm_193 = None
        conv2d_194 = torch.conv2d(
            input_64,
            l_self_modules_model_modules_21_modules_cv2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_64 = l_self_modules_model_modules_21_modules_cv2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_194 = torch.nn.functional.batch_norm(
            conv2d_194,
            l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_194 = l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv2_modules_1_modules_bn_parameters_bias_ = (None)
        input_65 = torch.nn.functional.silu(batch_norm_194, inplace=True)
        batch_norm_194 = None
        conv2d_195 = torch.conv2d(
            getitem_15,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_195 = torch.nn.functional.batch_norm(
            conv2d_195,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_195 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_156 = torch.nn.functional.silu(batch_norm_195, inplace=True)
        batch_norm_195 = None
        conv2d_196 = torch.conv2d(
            silu_156,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_196 = torch.nn.functional.batch_norm(
            conv2d_196,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_196 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_197 = torch.conv2d(
            silu_156,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_197 = torch.nn.functional.batch_norm(
            conv2d_197,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_197 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_117 = batch_norm_196 + batch_norm_197
        batch_norm_196 = batch_norm_197 = None
        add_118 = add_117 + 0
        add_117 = None
        silu_157 = torch.nn.functional.silu(add_118, inplace=True)
        add_118 = None
        conv2d_198 = torch.conv2d(
            silu_157,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_157 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_198 = torch.nn.functional.batch_norm(
            conv2d_198,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_198 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_158 = torch.nn.functional.silu(batch_norm_198, inplace=True)
        batch_norm_198 = None
        input_66 = silu_156 + silu_158
        silu_156 = silu_158 = None
        conv2d_199 = torch.conv2d(
            input_66,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_199 = torch.nn.functional.batch_norm(
            conv2d_199,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_199 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_200 = torch.conv2d(
            input_66,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_200 = torch.nn.functional.batch_norm(
            conv2d_200,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_200 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_120 = batch_norm_199 + batch_norm_200
        batch_norm_199 = batch_norm_200 = None
        add_121 = add_120 + 0
        add_120 = None
        silu_159 = torch.nn.functional.silu(add_121, inplace=True)
        add_121 = None
        conv2d_201 = torch.conv2d(
            silu_159,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_159 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_201 = torch.nn.functional.batch_norm(
            conv2d_201,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_201 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_160 = torch.nn.functional.silu(batch_norm_201, inplace=True)
        batch_norm_201 = None
        input_67 = input_66 + silu_160
        input_66 = silu_160 = None
        conv2d_202 = torch.conv2d(
            input_67,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_202 = torch.nn.functional.batch_norm(
            conv2d_202,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_202 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv1_modules_bn_parameters_bias_ = (None)
        conv2d_203 = torch.conv2d(
            input_67,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_203 = torch.nn.functional.batch_norm(
            conv2d_203,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_203 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv1_modules_conv2_modules_bn_parameters_bias_ = (None)
        add_123 = batch_norm_202 + batch_norm_203
        batch_norm_202 = batch_norm_203 = None
        add_124 = add_123 + 0
        add_123 = None
        silu_161 = torch.nn.functional.silu(add_124, inplace=True)
        add_124 = None
        conv2d_204 = torch.conv2d(
            silu_161,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_161 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_204 = torch.nn.functional.batch_norm(
            conv2d_204,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_204 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_162 = torch.nn.functional.silu(batch_norm_204, inplace=True)
        batch_norm_204 = None
        input_68 = input_67 + silu_162
        input_67 = silu_162 = None
        conv2d_205 = torch.conv2d(
            getitem_15,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_205 = torch.nn.functional.batch_norm(
            conv2d_205,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_205 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_163 = torch.nn.functional.silu(batch_norm_205, inplace=True)
        batch_norm_205 = None
        cat_25 = torch.cat((input_68, silu_163), 1)
        input_68 = silu_163 = None
        conv2d_206 = torch.conv2d(
            cat_25,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_25 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_206 = torch.nn.functional.batch_norm(
            conv2d_206,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_206 = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        input_69 = torch.nn.functional.silu(batch_norm_206, inplace=True)
        batch_norm_206 = None
        conv2d_207 = torch.conv2d(
            input_69,
            l_self_modules_model_modules_21_modules_cv3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_69 = l_self_modules_model_modules_21_modules_cv3_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_207 = torch.nn.functional.batch_norm(
            conv2d_207,
            l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_207 = l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_21_modules_cv3_modules_1_modules_bn_parameters_bias_ = (None)
        input_70 = torch.nn.functional.silu(batch_norm_207, inplace=True)
        batch_norm_207 = None
        cat_26 = torch.cat([getitem_14, getitem_15, input_65, input_70], 1)
        getitem_14 = getitem_15 = input_65 = input_70 = None
        conv2d_208 = torch.conv2d(
            cat_26,
            l_self_modules_model_modules_21_modules_cv4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_26 = (
            l_self_modules_model_modules_21_modules_cv4_modules_conv_parameters_weight_
        ) = None
        batch_norm_208 = torch.nn.functional.batch_norm(
            conv2d_208,
            l_self_modules_model_modules_21_modules_cv4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_21_modules_cv4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_21_modules_cv4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_21_modules_cv4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_208 = (
            l_self_modules_model_modules_21_modules_cv4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_21_modules_cv4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_21_modules_cv4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_21_modules_cv4_modules_bn_parameters_bias_
        ) = None
        x_26 = torch.nn.functional.silu(batch_norm_208, inplace=True)
        batch_norm_208 = None
        conv2d_209 = torch.conv2d(
            x_18,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_209 = torch.nn.functional.batch_norm(
            conv2d_209,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_209 = l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_71 = torch.nn.functional.silu(batch_norm_209, inplace=True)
        batch_norm_209 = None
        conv2d_210 = torch.conv2d(
            input_71,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_71 = l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_210 = torch.nn.functional.batch_norm(
            conv2d_210,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_210 = l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_72 = torch.nn.functional.silu(batch_norm_210, inplace=True)
        batch_norm_210 = None
        input_73 = torch.conv2d(
            input_72,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_72 = l_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_weight_ = l_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_bias_ = (None)
        conv2d_212 = torch.conv2d(
            x_18,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_211 = torch.nn.functional.batch_norm(
            conv2d_212,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_212 = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_74 = torch.nn.functional.silu(batch_norm_211, inplace=True)
        batch_norm_211 = None
        conv2d_213 = torch.conv2d(
            input_74,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_74 = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_212 = torch.nn.functional.batch_norm(
            conv2d_213,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_213 = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_75 = torch.nn.functional.silu(batch_norm_212, inplace=True)
        batch_norm_212 = None
        input_76 = torch.conv2d(
            input_75,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_75 = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_weight_ = l_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_bias_ = (None)
        xi = torch.cat((input_73, input_76), 1)
        input_73 = input_76 = None
        conv2d_215 = torch.conv2d(
            x_22,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_213 = torch.nn.functional.batch_norm(
            conv2d_215,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_215 = l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_77 = torch.nn.functional.silu(batch_norm_213, inplace=True)
        batch_norm_213 = None
        conv2d_216 = torch.conv2d(
            input_77,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_77 = l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_214 = torch.nn.functional.batch_norm(
            conv2d_216,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_216 = l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_78 = torch.nn.functional.silu(batch_norm_214, inplace=True)
        batch_norm_214 = None
        input_79 = torch.conv2d(
            input_78,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_78 = l_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_weight_ = l_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_bias_ = (None)
        conv2d_218 = torch.conv2d(
            x_22,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_215 = torch.nn.functional.batch_norm(
            conv2d_218,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_218 = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_80 = torch.nn.functional.silu(batch_norm_215, inplace=True)
        batch_norm_215 = None
        conv2d_219 = torch.conv2d(
            input_80,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_80 = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_216 = torch.nn.functional.batch_norm(
            conv2d_219,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_219 = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_81 = torch.nn.functional.silu(batch_norm_216, inplace=True)
        batch_norm_216 = None
        input_82 = torch.conv2d(
            input_81,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_81 = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_weight_ = l_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_bias_ = (None)
        xi_1 = torch.cat((input_79, input_82), 1)
        input_79 = input_82 = None
        conv2d_221 = torch.conv2d(
            x_26,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_217 = torch.nn.functional.batch_norm(
            conv2d_221,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_221 = l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        input_83 = torch.nn.functional.silu(batch_norm_217, inplace=True)
        batch_norm_217 = None
        conv2d_222 = torch.conv2d(
            input_83,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_83 = l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_218 = torch.nn.functional.batch_norm(
            conv2d_222,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_222 = l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        input_84 = torch.nn.functional.silu(batch_norm_218, inplace=True)
        batch_norm_218 = None
        input_85 = torch.conv2d(
            input_84,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_84 = l_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_weight_ = l_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_bias_ = (None)
        conv2d_224 = torch.conv2d(
            x_26,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_219 = torch.nn.functional.batch_norm(
            conv2d_224,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_224 = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        input_86 = torch.nn.functional.silu(batch_norm_219, inplace=True)
        batch_norm_219 = None
        conv2d_225 = torch.conv2d(
            input_86,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_86 = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_220 = torch.nn.functional.batch_norm(
            conv2d_225,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_225 = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        input_87 = torch.nn.functional.silu(batch_norm_220, inplace=True)
        batch_norm_220 = None
        input_88 = torch.conv2d(
            input_87,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_87 = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_weight_ = l_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_bias_ = (None)
        xi_2 = torch.cat((input_85, input_88), 1)
        input_85 = input_88 = None
        view = xi.view(1, 144, -1)
        view_1 = xi_1.view(1, 144, -1)
        view_2 = xi_2.view(1, 144, -1)
        x_cat = torch.cat([view, view_1, view_2], 2)
        view = view_1 = view_2 = None
        x_27 = l_self_modules_model_modules_22_stride[0]
        x_28 = l_self_modules_model_modules_22_stride[1]
        x_29 = l_self_modules_model_modules_22_stride[2]
        l_self_modules_model_modules_22_stride = None
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
        view_3 = stack.view(-1, 2)
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
        view_4 = stack_1.view(-1, 2)
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
        view_5 = stack_2.view(-1, 2)
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
        x_30 = torch.cat([view_3, view_4, view_5])
        view_3 = view_4 = view_5 = None
        x_31 = torch.cat([full, full_1, full_2])
        full = full_1 = full_2 = None
        transpose = x_30.transpose(0, 1)
        x_30 = None
        transpose_1 = x_31.transpose(0, 1)
        x_31 = None
        split = x_cat.split((64, 80), 1)
        x_cat = None
        box = split[0]
        cls = split[1]
        split = None
        view_6 = box.view(1, 4, 16, 8400)
        box = None
        transpose_2 = view_6.transpose(2, 1)
        view_6 = None
        softmax = transpose_2.softmax(1)
        transpose_2 = None
        conv2d_227 = torch.conv2d(
            softmax,
            l_self_modules_model_modules_22_modules_dfl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        softmax = (
            l_self_modules_model_modules_22_modules_dfl_modules_conv_parameters_weight_
        ) = None
        view_7 = conv2d_227.view(1, 4, 8400)
        conv2d_227 = None
        unsqueeze = transpose.unsqueeze(0)
        chunk_8 = view_7.chunk(2, 1)
        view_7 = None
        lt = chunk_8[0]
        rb = chunk_8[1]
        chunk_8 = None
        x1y1 = unsqueeze - lt
        lt = None
        x2y2 = unsqueeze + rb
        unsqueeze = rb = None
        add_133 = x1y1 + x2y2
        c_xy = add_133 / 2
        add_133 = None
        wh = x2y2 - x1y1
        x2y2 = x1y1 = None
        cat_33 = torch.cat((c_xy, wh), 1)
        c_xy = wh = None
        dbox = cat_33 * transpose_1
        cat_33 = None
        sigmoid = cls.sigmoid()
        cls = None
        y = torch.cat((dbox, sigmoid), 1)
        dbox = sigmoid = None
        return (y, xi, xi_1, xi_2, transpose_1, transpose)
