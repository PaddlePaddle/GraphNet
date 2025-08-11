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
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_proto_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_proto_modules_upsample_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_proto_modules_upsample_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_proto_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_proto_modules_cv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_cv4_modules_2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_proto_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_proto_modules_upsample_parameters_weight_ = L_self_modules_model_modules_23_modules_proto_modules_upsample_parameters_weight_
        l_self_modules_model_modules_23_modules_proto_modules_upsample_parameters_bias_ = L_self_modules_model_modules_23_modules_proto_modules_upsample_parameters_bias_
        l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_proto_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_proto_modules_cv3_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_2_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_2_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_2_parameters_bias_ = L_self_modules_model_modules_23_modules_cv4_modules_0_modules_2_parameters_bias_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_2_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_2_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_2_parameters_bias_ = L_self_modules_model_modules_23_modules_cv4_modules_1_modules_2_parameters_bias_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_2_parameters_weight_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_2_parameters_weight_
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_2_parameters_bias_ = L_self_modules_model_modules_23_modules_cv4_modules_2_modules_2_parameters_bias_
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
            0.03,
            0.001,
        )
        conv2d_10 = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_10 = torch.nn.functional.silu(batch_norm_10, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_11 = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_11 = torch.nn.functional.silu(batch_norm_11, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_12 = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_12 = torch.nn.functional.silu(batch_norm_12, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_13 = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_13 = torch.nn.functional.silu(batch_norm_13, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_14 = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_14 = torch.nn.functional.silu(batch_norm_14, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_15 = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_15 = torch.nn.functional.silu(batch_norm_15, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_16 = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_16 = torch.nn.functional.silu(batch_norm_16, inplace=True)
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
            0.03,
            0.001,
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
        x_2 = torch.nn.functional.silu(batch_norm_17, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_18 = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_3_modules_bn_parameters_bias_ = None
        x_3 = torch.nn.functional.silu(batch_norm_18, inplace=True)
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
            0.03,
            0.001,
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
        silu_19 = torch.nn.functional.silu(batch_norm_19, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_20 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_20 = torch.nn.functional.silu(batch_norm_20, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_21 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_21 = torch.nn.functional.silu(batch_norm_21, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_22 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_22 = torch.nn.functional.silu(batch_norm_22, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_23 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_23 = torch.nn.functional.silu(batch_norm_23, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_24 = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_24 = torch.nn.functional.silu(batch_norm_24, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_25 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_25 = torch.nn.functional.silu(batch_norm_25, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_26 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_26 = torch.nn.functional.silu(batch_norm_26, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_27 = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_27 = torch.nn.functional.silu(batch_norm_27, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_28 = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_28 = torch.nn.functional.silu(batch_norm_28, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_29 = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_29 = torch.nn.functional.silu(batch_norm_29, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_30 = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_30 = torch.nn.functional.silu(batch_norm_30, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_31 = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_31 = torch.nn.functional.silu(batch_norm_31, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_32 = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_32 = torch.nn.functional.silu(batch_norm_32, inplace=True)
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
            0.03,
            0.001,
        )
        conv2d_33 = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_33 = torch.nn.functional.silu(batch_norm_33, inplace=True)
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
            0.03,
            0.001,
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
        x_4 = torch.nn.functional.silu(batch_norm_34, inplace=True)
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
        l_self_modules_model_modules_5_modules_conv_parameters_weight_ = None
        batch_norm_35 = torch.nn.functional.batch_norm(
            conv2d_35,
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_5_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_35 = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_5_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_5_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.silu(batch_norm_35, inplace=True)
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
        x_5 = (
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
        silu_36 = torch.nn.functional.silu(batch_norm_36, inplace=True)
        batch_norm_36 = None
        chunk_2 = silu_36.chunk(2, 1)
        silu_36 = None
        getitem_4 = chunk_2[0]
        getitem_5 = chunk_2[1]
        chunk_2 = None
        conv2d_37 = torch.conv2d(
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
        batch_norm_37 = torch.nn.functional.batch_norm(
            conv2d_37,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_37 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_37 = torch.nn.functional.silu(batch_norm_37, inplace=True)
        batch_norm_37 = None
        conv2d_38 = torch.conv2d(
            silu_37,
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
        batch_norm_38 = torch.nn.functional.batch_norm(
            conv2d_38,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_38 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_38 = torch.nn.functional.silu(batch_norm_38, inplace=True)
        batch_norm_38 = None
        conv2d_39 = torch.conv2d(
            silu_38,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_38 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_39 = torch.nn.functional.batch_norm(
            conv2d_39,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_39 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_39 = torch.nn.functional.silu(batch_norm_39, inplace=True)
        batch_norm_39 = None
        input_9 = silu_37 + silu_39
        silu_37 = silu_39 = None
        conv2d_40 = torch.conv2d(
            input_9,
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
        batch_norm_40 = torch.nn.functional.batch_norm(
            conv2d_40,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_40 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_40 = torch.nn.functional.silu(batch_norm_40, inplace=True)
        batch_norm_40 = None
        conv2d_41 = torch.conv2d(
            silu_40,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_40 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_41 = torch.nn.functional.batch_norm(
            conv2d_41,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_41 = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_41 = torch.nn.functional.silu(batch_norm_41, inplace=True)
        batch_norm_41 = None
        input_10 = input_9 + silu_41
        input_9 = silu_41 = None
        conv2d_42 = torch.conv2d(
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
        batch_norm_42 = torch.nn.functional.batch_norm(
            conv2d_42,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_42 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_42 = torch.nn.functional.silu(batch_norm_42, inplace=True)
        batch_norm_42 = None
        cat_6 = torch.cat((input_10, silu_42), 1)
        input_10 = silu_42 = None
        conv2d_43 = torch.conv2d(
            cat_6,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_6 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_43 = torch.nn.functional.batch_norm(
            conv2d_43,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_43 = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_43 = torch.nn.functional.silu(batch_norm_43, inplace=True)
        batch_norm_43 = None
        conv2d_44 = torch.conv2d(
            getitem_5,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_44 = torch.nn.functional.batch_norm(
            conv2d_44,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_44 = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_44 = torch.nn.functional.silu(batch_norm_44, inplace=True)
        batch_norm_44 = None
        conv2d_45 = torch.conv2d(
            silu_44,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_45 = torch.nn.functional.batch_norm(
            conv2d_45,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_45 = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_45 = torch.nn.functional.silu(batch_norm_45, inplace=True)
        batch_norm_45 = None
        conv2d_46 = torch.conv2d(
            silu_45,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_45 = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_46 = torch.nn.functional.batch_norm(
            conv2d_46,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_46 = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_46 = torch.nn.functional.silu(batch_norm_46, inplace=True)
        batch_norm_46 = None
        input_11 = silu_44 + silu_46
        silu_44 = silu_46 = None
        conv2d_47 = torch.conv2d(
            input_11,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_47 = torch.nn.functional.batch_norm(
            conv2d_47,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_47 = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_47 = torch.nn.functional.silu(batch_norm_47, inplace=True)
        batch_norm_47 = None
        conv2d_48 = torch.conv2d(
            silu_47,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_47 = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_48 = torch.nn.functional.batch_norm(
            conv2d_48,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_48 = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_48 = torch.nn.functional.silu(batch_norm_48, inplace=True)
        batch_norm_48 = None
        input_12 = input_11 + silu_48
        input_11 = silu_48 = None
        conv2d_49 = torch.conv2d(
            getitem_5,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_49 = torch.nn.functional.batch_norm(
            conv2d_49,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_49 = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_49 = torch.nn.functional.silu(batch_norm_49, inplace=True)
        batch_norm_49 = None
        cat_7 = torch.cat((input_12, silu_49), 1)
        input_12 = silu_49 = None
        conv2d_50 = torch.conv2d(
            cat_7,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_7 = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_50 = torch.nn.functional.batch_norm(
            conv2d_50,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_50 = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_50 = torch.nn.functional.silu(batch_norm_50, inplace=True)
        batch_norm_50 = None
        cat_8 = torch.cat([getitem_4, getitem_5, silu_43, silu_50], 1)
        getitem_4 = getitem_5 = silu_43 = silu_50 = None
        conv2d_51 = torch.conv2d(
            cat_8,
            l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_8 = (
            l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_51 = torch.nn.functional.batch_norm(
            conv2d_51,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_51 = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_6 = torch.nn.functional.silu(batch_norm_51, inplace=True)
        batch_norm_51 = None
        conv2d_52 = torch.conv2d(
            x_6,
            l_self_modules_model_modules_7_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_7_modules_conv_parameters_weight_ = None
        batch_norm_52 = torch.nn.functional.batch_norm(
            conv2d_52,
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_7_modules_bn_parameters_weight_,
            l_self_modules_model_modules_7_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_52 = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_7_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_7_modules_bn_parameters_bias_ = None
        x_7 = torch.nn.functional.silu(batch_norm_52, inplace=True)
        batch_norm_52 = None
        conv2d_53 = torch.conv2d(
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
        batch_norm_53 = torch.nn.functional.batch_norm(
            conv2d_53,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_53 = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_53 = torch.nn.functional.silu(batch_norm_53, inplace=True)
        batch_norm_53 = None
        chunk_3 = silu_53.chunk(2, 1)
        silu_53 = None
        getitem_6 = chunk_3[0]
        getitem_7 = chunk_3[1]
        chunk_3 = None
        conv2d_54 = torch.conv2d(
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
        batch_norm_54 = torch.nn.functional.batch_norm(
            conv2d_54,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_54 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_54 = torch.nn.functional.silu(batch_norm_54, inplace=True)
        batch_norm_54 = None
        conv2d_55 = torch.conv2d(
            silu_54,
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
        batch_norm_55 = torch.nn.functional.batch_norm(
            conv2d_55,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_55 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_55 = torch.nn.functional.silu(batch_norm_55, inplace=True)
        batch_norm_55 = None
        conv2d_56 = torch.conv2d(
            silu_55,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_55 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_56 = torch.nn.functional.batch_norm(
            conv2d_56,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_56 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_56 = torch.nn.functional.silu(batch_norm_56, inplace=True)
        batch_norm_56 = None
        input_13 = silu_54 + silu_56
        silu_54 = silu_56 = None
        conv2d_57 = torch.conv2d(
            input_13,
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
        batch_norm_57 = torch.nn.functional.batch_norm(
            conv2d_57,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_57 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_57 = torch.nn.functional.silu(batch_norm_57, inplace=True)
        batch_norm_57 = None
        conv2d_58 = torch.conv2d(
            silu_57,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_57 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_58 = torch.nn.functional.batch_norm(
            conv2d_58,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_58 = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_58 = torch.nn.functional.silu(batch_norm_58, inplace=True)
        batch_norm_58 = None
        input_14 = input_13 + silu_58
        input_13 = silu_58 = None
        conv2d_59 = torch.conv2d(
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
        batch_norm_59 = torch.nn.functional.batch_norm(
            conv2d_59,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_59 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_59 = torch.nn.functional.silu(batch_norm_59, inplace=True)
        batch_norm_59 = None
        cat_9 = torch.cat((input_14, silu_59), 1)
        input_14 = silu_59 = None
        conv2d_60 = torch.conv2d(
            cat_9,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_9 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_60 = torch.nn.functional.batch_norm(
            conv2d_60,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_60 = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_60 = torch.nn.functional.silu(batch_norm_60, inplace=True)
        batch_norm_60 = None
        conv2d_61 = torch.conv2d(
            getitem_7,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_61 = torch.nn.functional.batch_norm(
            conv2d_61,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_61 = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_61 = torch.nn.functional.silu(batch_norm_61, inplace=True)
        batch_norm_61 = None
        conv2d_62 = torch.conv2d(
            silu_61,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_62 = torch.nn.functional.batch_norm(
            conv2d_62,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_62 = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_62 = torch.nn.functional.silu(batch_norm_62, inplace=True)
        batch_norm_62 = None
        conv2d_63 = torch.conv2d(
            silu_62,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_62 = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_63 = torch.nn.functional.batch_norm(
            conv2d_63,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_63 = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_63 = torch.nn.functional.silu(batch_norm_63, inplace=True)
        batch_norm_63 = None
        input_15 = silu_61 + silu_63
        silu_61 = silu_63 = None
        conv2d_64 = torch.conv2d(
            input_15,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_64 = torch.nn.functional.batch_norm(
            conv2d_64,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_64 = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_64 = torch.nn.functional.silu(batch_norm_64, inplace=True)
        batch_norm_64 = None
        conv2d_65 = torch.conv2d(
            silu_64,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_64 = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_65 = torch.nn.functional.batch_norm(
            conv2d_65,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_65 = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_65 = torch.nn.functional.silu(batch_norm_65, inplace=True)
        batch_norm_65 = None
        input_16 = input_15 + silu_65
        input_15 = silu_65 = None
        conv2d_66 = torch.conv2d(
            getitem_7,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_66 = torch.nn.functional.batch_norm(
            conv2d_66,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_66 = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_66 = torch.nn.functional.silu(batch_norm_66, inplace=True)
        batch_norm_66 = None
        cat_10 = torch.cat((input_16, silu_66), 1)
        input_16 = silu_66 = None
        conv2d_67 = torch.conv2d(
            cat_10,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_10 = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_67 = torch.nn.functional.batch_norm(
            conv2d_67,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_67 = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_67 = torch.nn.functional.silu(batch_norm_67, inplace=True)
        batch_norm_67 = None
        cat_11 = torch.cat([getitem_6, getitem_7, silu_60, silu_67], 1)
        getitem_6 = getitem_7 = silu_60 = silu_67 = None
        conv2d_68 = torch.conv2d(
            cat_11,
            l_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_11 = (
            l_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_68 = torch.nn.functional.batch_norm(
            conv2d_68,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_68 = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.silu(batch_norm_68, inplace=True)
        batch_norm_68 = None
        conv2d_69 = torch.conv2d(
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
        batch_norm_69 = torch.nn.functional.batch_norm(
            conv2d_69,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_69 = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_69 = torch.nn.functional.silu(batch_norm_69, inplace=True)
        batch_norm_69 = None
        max_pool2d = torch.nn.functional.max_pool2d(
            silu_69, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_1 = torch.nn.functional.max_pool2d(
            silu_69, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_2 = torch.nn.functional.max_pool2d(
            silu_69, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        cat_12 = torch.cat([silu_69, max_pool2d, max_pool2d_1, max_pool2d_2], 1)
        silu_69 = max_pool2d = max_pool2d_1 = max_pool2d_2 = None
        conv2d_70 = torch.conv2d(
            cat_12,
            l_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_12 = (
            l_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_70 = torch.nn.functional.batch_norm(
            conv2d_70,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_70 = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_9 = torch.nn.functional.silu(batch_norm_70, inplace=True)
        batch_norm_70 = None
        conv2d_71 = torch.conv2d(
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
        batch_norm_71 = torch.nn.functional.batch_norm(
            conv2d_71,
            l_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_71 = (
            l_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_71 = torch.nn.functional.silu(batch_norm_71, inplace=True)
        batch_norm_71 = None
        split = silu_71.split((384, 384), dim=1)
        silu_71 = None
        a = split[0]
        b = split[1]
        split = None
        conv2d_72 = torch.conv2d(
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
            conv2d_72,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_72 = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        view = qkv.view(1, 6, 128, 400)
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
        view_1 = matmul_1.view(1, 384, 20, 20)
        matmul_1 = None
        reshape = v.reshape(1, 384, 20, 20)
        v = None
        conv2d_73 = torch.conv2d(
            reshape,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        reshape = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_73 = torch.nn.functional.batch_norm(
            conv2d_73,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_73 = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_10 = view_1 + batch_norm_73
        view_1 = batch_norm_73 = None
        conv2d_74 = torch.conv2d(
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
            conv2d_74,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_74 = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_12 = b + x_11
        b = x_11 = None
        conv2d_75 = torch.conv2d(
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
        batch_norm_75 = torch.nn.functional.batch_norm(
            conv2d_75,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_75 = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_ = (None)
        input_17 = torch.nn.functional.silu(batch_norm_75, inplace=True)
        batch_norm_75 = None
        conv2d_76 = torch.conv2d(
            input_17,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_17 = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_ = (None)
        input_18 = torch.nn.functional.batch_norm(
            conv2d_76,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_76 = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_ = (None)
        x_13 = x_12 + input_18
        x_12 = input_18 = None
        conv2d_77 = torch.conv2d(
            x_13,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        qkv_1 = torch.nn.functional.batch_norm(
            conv2d_77,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_77 = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (None)
        view_2 = qkv_1.view(1, 6, 128, 400)
        qkv_1 = None
        split_2 = view_2.split([32, 32, 64], dim=2)
        view_2 = None
        q_1 = split_2[0]
        k_1 = split_2[1]
        v_1 = split_2[2]
        split_2 = None
        transpose_2 = q_1.transpose(-2, -1)
        q_1 = None
        matmul_2 = transpose_2 @ k_1
        transpose_2 = k_1 = None
        attn_2 = matmul_2 * 0.1767766952966369
        matmul_2 = None
        attn_3 = attn_2.softmax(dim=-1)
        attn_2 = None
        transpose_3 = attn_3.transpose(-2, -1)
        attn_3 = None
        matmul_3 = v_1 @ transpose_3
        transpose_3 = None
        view_3 = matmul_3.view(1, 384, 20, 20)
        matmul_3 = None
        reshape_1 = v_1.reshape(1, 384, 20, 20)
        v_1 = None
        conv2d_78 = torch.conv2d(
            reshape_1,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        reshape_1 = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        batch_norm_78 = torch.nn.functional.batch_norm(
            conv2d_78,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_78 = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (None)
        x_14 = view_3 + batch_norm_78
        view_3 = batch_norm_78 = None
        conv2d_79 = torch.conv2d(
            x_14,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            conv2d_79,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_79 = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (None)
        x_16 = x_13 + x_15
        x_13 = x_15 = None
        conv2d_80 = torch.conv2d(
            x_16,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_80 = torch.nn.functional.batch_norm(
            conv2d_80,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_80 = l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_0_modules_bn_parameters_bias_ = (None)
        input_19 = torch.nn.functional.silu(batch_norm_80, inplace=True)
        batch_norm_80 = None
        conv2d_81 = torch.conv2d(
            input_19,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_19 = l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_conv_parameters_weight_ = (None)
        input_20 = torch.nn.functional.batch_norm(
            conv2d_81,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_81 = l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_10_modules_m_modules_1_modules_ffn_modules_1_modules_bn_parameters_bias_ = (None)
        x_17 = x_16 + input_20
        x_16 = input_20 = None
        cat_13 = torch.cat((a, x_17), 1)
        a = x_17 = None
        conv2d_82 = torch.conv2d(
            cat_13,
            l_self_modules_model_modules_10_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_13 = (
            l_self_modules_model_modules_10_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_82 = torch.nn.functional.batch_norm(
            conv2d_82,
            l_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_82 = (
            l_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_18 = torch.nn.functional.silu(batch_norm_82, inplace=True)
        batch_norm_82 = None
        x_19 = torch.nn.functional.interpolate(
            x_18, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        x_20 = torch.cat([x_19, x_6], 1)
        x_19 = x_6 = None
        conv2d_83 = torch.conv2d(
            x_20,
            l_self_modules_model_modules_13_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = (
            l_self_modules_model_modules_13_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_83 = torch.nn.functional.batch_norm(
            conv2d_83,
            l_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_83 = (
            l_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_13_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_13_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_75 = torch.nn.functional.silu(batch_norm_83, inplace=True)
        batch_norm_83 = None
        chunk_4 = silu_75.chunk(2, 1)
        silu_75 = None
        getitem_16 = chunk_4[0]
        getitem_17 = chunk_4[1]
        chunk_4 = None
        conv2d_84 = torch.conv2d(
            getitem_17,
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
        batch_norm_84 = torch.nn.functional.batch_norm(
            conv2d_84,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_84 = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_76 = torch.nn.functional.silu(batch_norm_84, inplace=True)
        batch_norm_84 = None
        conv2d_85 = torch.conv2d(
            silu_76,
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
        batch_norm_85 = torch.nn.functional.batch_norm(
            conv2d_85,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_85 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_77 = torch.nn.functional.silu(batch_norm_85, inplace=True)
        batch_norm_85 = None
        conv2d_86 = torch.conv2d(
            silu_77,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_77 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_86 = torch.nn.functional.batch_norm(
            conv2d_86,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_86 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_78 = torch.nn.functional.silu(batch_norm_86, inplace=True)
        batch_norm_86 = None
        input_21 = silu_76 + silu_78
        silu_76 = silu_78 = None
        conv2d_87 = torch.conv2d(
            input_21,
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
        batch_norm_87 = torch.nn.functional.batch_norm(
            conv2d_87,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_87 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_79 = torch.nn.functional.silu(batch_norm_87, inplace=True)
        batch_norm_87 = None
        conv2d_88 = torch.conv2d(
            silu_79,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_79 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_88 = torch.nn.functional.batch_norm(
            conv2d_88,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_88 = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_80 = torch.nn.functional.silu(batch_norm_88, inplace=True)
        batch_norm_88 = None
        input_22 = input_21 + silu_80
        input_21 = silu_80 = None
        conv2d_89 = torch.conv2d(
            getitem_17,
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
        batch_norm_89 = torch.nn.functional.batch_norm(
            conv2d_89,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_89 = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_81 = torch.nn.functional.silu(batch_norm_89, inplace=True)
        batch_norm_89 = None
        cat_15 = torch.cat((input_22, silu_81), 1)
        input_22 = silu_81 = None
        conv2d_90 = torch.conv2d(
            cat_15,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_15 = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_90 = torch.nn.functional.batch_norm(
            conv2d_90,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_90 = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_82 = torch.nn.functional.silu(batch_norm_90, inplace=True)
        batch_norm_90 = None
        conv2d_91 = torch.conv2d(
            getitem_17,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_91 = torch.nn.functional.batch_norm(
            conv2d_91,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_91 = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_83 = torch.nn.functional.silu(batch_norm_91, inplace=True)
        batch_norm_91 = None
        conv2d_92 = torch.conv2d(
            silu_83,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_92 = torch.nn.functional.batch_norm(
            conv2d_92,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_92 = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_84 = torch.nn.functional.silu(batch_norm_92, inplace=True)
        batch_norm_92 = None
        conv2d_93 = torch.conv2d(
            silu_84,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_84 = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_93 = torch.nn.functional.batch_norm(
            conv2d_93,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_93 = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_85 = torch.nn.functional.silu(batch_norm_93, inplace=True)
        batch_norm_93 = None
        input_23 = silu_83 + silu_85
        silu_83 = silu_85 = None
        conv2d_94 = torch.conv2d(
            input_23,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_94 = torch.nn.functional.batch_norm(
            conv2d_94,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_94 = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_86 = torch.nn.functional.silu(batch_norm_94, inplace=True)
        batch_norm_94 = None
        conv2d_95 = torch.conv2d(
            silu_86,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_86 = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_95 = torch.nn.functional.batch_norm(
            conv2d_95,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_95 = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_87 = torch.nn.functional.silu(batch_norm_95, inplace=True)
        batch_norm_95 = None
        input_24 = input_23 + silu_87
        input_23 = silu_87 = None
        conv2d_96 = torch.conv2d(
            getitem_17,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_96 = torch.nn.functional.batch_norm(
            conv2d_96,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_96 = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_88 = torch.nn.functional.silu(batch_norm_96, inplace=True)
        batch_norm_96 = None
        cat_16 = torch.cat((input_24, silu_88), 1)
        input_24 = silu_88 = None
        conv2d_97 = torch.conv2d(
            cat_16,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_16 = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_97 = torch.nn.functional.batch_norm(
            conv2d_97,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_97 = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_13_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_89 = torch.nn.functional.silu(batch_norm_97, inplace=True)
        batch_norm_97 = None
        cat_17 = torch.cat([getitem_16, getitem_17, silu_82, silu_89], 1)
        getitem_16 = getitem_17 = silu_82 = silu_89 = None
        conv2d_98 = torch.conv2d(
            cat_17,
            l_self_modules_model_modules_13_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_17 = (
            l_self_modules_model_modules_13_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_98 = torch.nn.functional.batch_norm(
            conv2d_98,
            l_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_98 = (
            l_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_13_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_13_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_21 = torch.nn.functional.silu(batch_norm_98, inplace=True)
        batch_norm_98 = None
        x_22 = torch.nn.functional.interpolate(
            x_21, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        x_23 = torch.cat([x_22, x_4], 1)
        x_22 = x_4 = None
        conv2d_99 = torch.conv2d(
            x_23,
            l_self_modules_model_modules_16_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = (
            l_self_modules_model_modules_16_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_99 = torch.nn.functional.batch_norm(
            conv2d_99,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_99 = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_16_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_91 = torch.nn.functional.silu(batch_norm_99, inplace=True)
        batch_norm_99 = None
        chunk_5 = silu_91.chunk(2, 1)
        silu_91 = None
        getitem_18 = chunk_5[0]
        getitem_19 = chunk_5[1]
        chunk_5 = None
        conv2d_100 = torch.conv2d(
            getitem_19,
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
        batch_norm_100 = torch.nn.functional.batch_norm(
            conv2d_100,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_100 = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_92 = torch.nn.functional.silu(batch_norm_100, inplace=True)
        batch_norm_100 = None
        conv2d_101 = torch.conv2d(
            silu_92,
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
        batch_norm_101 = torch.nn.functional.batch_norm(
            conv2d_101,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_101 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_93 = torch.nn.functional.silu(batch_norm_101, inplace=True)
        batch_norm_101 = None
        conv2d_102 = torch.conv2d(
            silu_93,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_93 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_102 = torch.nn.functional.batch_norm(
            conv2d_102,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_102 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_94 = torch.nn.functional.silu(batch_norm_102, inplace=True)
        batch_norm_102 = None
        input_25 = silu_92 + silu_94
        silu_92 = silu_94 = None
        conv2d_103 = torch.conv2d(
            input_25,
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
        batch_norm_103 = torch.nn.functional.batch_norm(
            conv2d_103,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_103 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_95 = torch.nn.functional.silu(batch_norm_103, inplace=True)
        batch_norm_103 = None
        conv2d_104 = torch.conv2d(
            silu_95,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_95 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_104 = torch.nn.functional.batch_norm(
            conv2d_104,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_104 = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_96 = torch.nn.functional.silu(batch_norm_104, inplace=True)
        batch_norm_104 = None
        input_26 = input_25 + silu_96
        input_25 = silu_96 = None
        conv2d_105 = torch.conv2d(
            getitem_19,
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
        batch_norm_105 = torch.nn.functional.batch_norm(
            conv2d_105,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_105 = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_97 = torch.nn.functional.silu(batch_norm_105, inplace=True)
        batch_norm_105 = None
        cat_19 = torch.cat((input_26, silu_97), 1)
        input_26 = silu_97 = None
        conv2d_106 = torch.conv2d(
            cat_19,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_19 = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_106 = torch.nn.functional.batch_norm(
            conv2d_106,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_106 = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_98 = torch.nn.functional.silu(batch_norm_106, inplace=True)
        batch_norm_106 = None
        conv2d_107 = torch.conv2d(
            getitem_19,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_107 = torch.nn.functional.batch_norm(
            conv2d_107,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_107 = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_99 = torch.nn.functional.silu(batch_norm_107, inplace=True)
        batch_norm_107 = None
        conv2d_108 = torch.conv2d(
            silu_99,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_108 = torch.nn.functional.batch_norm(
            conv2d_108,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_108 = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_100 = torch.nn.functional.silu(batch_norm_108, inplace=True)
        batch_norm_108 = None
        conv2d_109 = torch.conv2d(
            silu_100,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_100 = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_109 = torch.nn.functional.batch_norm(
            conv2d_109,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_109 = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_101 = torch.nn.functional.silu(batch_norm_109, inplace=True)
        batch_norm_109 = None
        input_27 = silu_99 + silu_101
        silu_99 = silu_101 = None
        conv2d_110 = torch.conv2d(
            input_27,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_110 = torch.nn.functional.batch_norm(
            conv2d_110,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_110 = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_102 = torch.nn.functional.silu(batch_norm_110, inplace=True)
        batch_norm_110 = None
        conv2d_111 = torch.conv2d(
            silu_102,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_102 = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_111 = torch.nn.functional.batch_norm(
            conv2d_111,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_111 = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_103 = torch.nn.functional.silu(batch_norm_111, inplace=True)
        batch_norm_111 = None
        input_28 = input_27 + silu_103
        input_27 = silu_103 = None
        conv2d_112 = torch.conv2d(
            getitem_19,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_112 = torch.nn.functional.batch_norm(
            conv2d_112,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_112 = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_104 = torch.nn.functional.silu(batch_norm_112, inplace=True)
        batch_norm_112 = None
        cat_20 = torch.cat((input_28, silu_104), 1)
        input_28 = silu_104 = None
        conv2d_113 = torch.conv2d(
            cat_20,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_20 = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_113 = torch.nn.functional.batch_norm(
            conv2d_113,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_113 = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_16_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_105 = torch.nn.functional.silu(batch_norm_113, inplace=True)
        batch_norm_113 = None
        cat_21 = torch.cat([getitem_18, getitem_19, silu_98, silu_105], 1)
        getitem_18 = getitem_19 = silu_98 = silu_105 = None
        conv2d_114 = torch.conv2d(
            cat_21,
            l_self_modules_model_modules_16_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_21 = (
            l_self_modules_model_modules_16_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_114 = torch.nn.functional.batch_norm(
            conv2d_114,
            l_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_114 = (
            l_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_16_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_16_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_24 = torch.nn.functional.silu(batch_norm_114, inplace=True)
        batch_norm_114 = None
        conv2d_115 = torch.conv2d(
            x_24,
            l_self_modules_model_modules_17_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_17_modules_conv_parameters_weight_ = None
        batch_norm_115 = torch.nn.functional.batch_norm(
            conv2d_115,
            l_self_modules_model_modules_17_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_17_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_17_modules_bn_parameters_weight_,
            l_self_modules_model_modules_17_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_115 = (
            l_self_modules_model_modules_17_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_17_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_17_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_17_modules_bn_parameters_bias_ = None
        x_25 = torch.nn.functional.silu(batch_norm_115, inplace=True)
        batch_norm_115 = None
        x_26 = torch.cat([x_25, x_21], 1)
        x_25 = x_21 = None
        conv2d_116 = torch.conv2d(
            x_26,
            l_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = (
            l_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_116 = torch.nn.functional.batch_norm(
            conv2d_116,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_116 = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_108 = torch.nn.functional.silu(batch_norm_116, inplace=True)
        batch_norm_116 = None
        chunk_6 = silu_108.chunk(2, 1)
        silu_108 = None
        getitem_20 = chunk_6[0]
        getitem_21 = chunk_6[1]
        chunk_6 = None
        conv2d_117 = torch.conv2d(
            getitem_21,
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
        batch_norm_117 = torch.nn.functional.batch_norm(
            conv2d_117,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_117 = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_109 = torch.nn.functional.silu(batch_norm_117, inplace=True)
        batch_norm_117 = None
        conv2d_118 = torch.conv2d(
            silu_109,
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
        batch_norm_118 = torch.nn.functional.batch_norm(
            conv2d_118,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_118 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_110 = torch.nn.functional.silu(batch_norm_118, inplace=True)
        batch_norm_118 = None
        conv2d_119 = torch.conv2d(
            silu_110,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_110 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_119 = torch.nn.functional.batch_norm(
            conv2d_119,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_119 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_111 = torch.nn.functional.silu(batch_norm_119, inplace=True)
        batch_norm_119 = None
        input_29 = silu_109 + silu_111
        silu_109 = silu_111 = None
        conv2d_120 = torch.conv2d(
            input_29,
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
        batch_norm_120 = torch.nn.functional.batch_norm(
            conv2d_120,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_120 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_112 = torch.nn.functional.silu(batch_norm_120, inplace=True)
        batch_norm_120 = None
        conv2d_121 = torch.conv2d(
            silu_112,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_112 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_121 = torch.nn.functional.batch_norm(
            conv2d_121,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_121 = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_113 = torch.nn.functional.silu(batch_norm_121, inplace=True)
        batch_norm_121 = None
        input_30 = input_29 + silu_113
        input_29 = silu_113 = None
        conv2d_122 = torch.conv2d(
            getitem_21,
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
        batch_norm_122 = torch.nn.functional.batch_norm(
            conv2d_122,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_122 = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_114 = torch.nn.functional.silu(batch_norm_122, inplace=True)
        batch_norm_122 = None
        cat_23 = torch.cat((input_30, silu_114), 1)
        input_30 = silu_114 = None
        conv2d_123 = torch.conv2d(
            cat_23,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_23 = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_123 = torch.nn.functional.batch_norm(
            conv2d_123,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_123 = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_115 = torch.nn.functional.silu(batch_norm_123, inplace=True)
        batch_norm_123 = None
        conv2d_124 = torch.conv2d(
            getitem_21,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_124 = torch.nn.functional.batch_norm(
            conv2d_124,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_124 = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_116 = torch.nn.functional.silu(batch_norm_124, inplace=True)
        batch_norm_124 = None
        conv2d_125 = torch.conv2d(
            silu_116,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_125 = torch.nn.functional.batch_norm(
            conv2d_125,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_125 = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_117 = torch.nn.functional.silu(batch_norm_125, inplace=True)
        batch_norm_125 = None
        conv2d_126 = torch.conv2d(
            silu_117,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_117 = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_126 = torch.nn.functional.batch_norm(
            conv2d_126,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_126 = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_118 = torch.nn.functional.silu(batch_norm_126, inplace=True)
        batch_norm_126 = None
        input_31 = silu_116 + silu_118
        silu_116 = silu_118 = None
        conv2d_127 = torch.conv2d(
            input_31,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_127 = torch.nn.functional.batch_norm(
            conv2d_127,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_127 = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_119 = torch.nn.functional.silu(batch_norm_127, inplace=True)
        batch_norm_127 = None
        conv2d_128 = torch.conv2d(
            silu_119,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_119 = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_128 = torch.nn.functional.batch_norm(
            conv2d_128,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_128 = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_120 = torch.nn.functional.silu(batch_norm_128, inplace=True)
        batch_norm_128 = None
        input_32 = input_31 + silu_120
        input_31 = silu_120 = None
        conv2d_129 = torch.conv2d(
            getitem_21,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_129 = torch.nn.functional.batch_norm(
            conv2d_129,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_129 = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_121 = torch.nn.functional.silu(batch_norm_129, inplace=True)
        batch_norm_129 = None
        cat_24 = torch.cat((input_32, silu_121), 1)
        input_32 = silu_121 = None
        conv2d_130 = torch.conv2d(
            cat_24,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_24 = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_130 = torch.nn.functional.batch_norm(
            conv2d_130,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_130 = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_19_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_122 = torch.nn.functional.silu(batch_norm_130, inplace=True)
        batch_norm_130 = None
        cat_25 = torch.cat([getitem_20, getitem_21, silu_115, silu_122], 1)
        getitem_20 = getitem_21 = silu_115 = silu_122 = None
        conv2d_131 = torch.conv2d(
            cat_25,
            l_self_modules_model_modules_19_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_25 = (
            l_self_modules_model_modules_19_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_131 = torch.nn.functional.batch_norm(
            conv2d_131,
            l_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_131 = (
            l_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_27 = torch.nn.functional.silu(batch_norm_131, inplace=True)
        batch_norm_131 = None
        conv2d_132 = torch.conv2d(
            x_27,
            l_self_modules_model_modules_20_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_20_modules_conv_parameters_weight_ = None
        batch_norm_132 = torch.nn.functional.batch_norm(
            conv2d_132,
            l_self_modules_model_modules_20_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_20_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_20_modules_bn_parameters_weight_,
            l_self_modules_model_modules_20_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_132 = (
            l_self_modules_model_modules_20_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_20_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_20_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_20_modules_bn_parameters_bias_ = None
        x_28 = torch.nn.functional.silu(batch_norm_132, inplace=True)
        batch_norm_132 = None
        x_29 = torch.cat([x_28, x_18], 1)
        x_28 = x_18 = None
        conv2d_133 = torch.conv2d(
            x_29,
            l_self_modules_model_modules_22_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_29 = (
            l_self_modules_model_modules_22_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_133 = torch.nn.functional.batch_norm(
            conv2d_133,
            l_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_133 = (
            l_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_22_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_22_modules_cv1_modules_bn_parameters_bias_
        ) = None
        silu_125 = torch.nn.functional.silu(batch_norm_133, inplace=True)
        batch_norm_133 = None
        chunk_7 = silu_125.chunk(2, 1)
        silu_125 = None
        getitem_22 = chunk_7[0]
        getitem_23 = chunk_7[1]
        chunk_7 = None
        conv2d_134 = torch.conv2d(
            getitem_23,
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
        batch_norm_134 = torch.nn.functional.batch_norm(
            conv2d_134,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_134 = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_126 = torch.nn.functional.silu(batch_norm_134, inplace=True)
        batch_norm_134 = None
        conv2d_135 = torch.conv2d(
            silu_126,
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
        batch_norm_135 = torch.nn.functional.batch_norm(
            conv2d_135,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_135 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_127 = torch.nn.functional.silu(batch_norm_135, inplace=True)
        batch_norm_135 = None
        conv2d_136 = torch.conv2d(
            silu_127,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_127 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_136 = torch.nn.functional.batch_norm(
            conv2d_136,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_136 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_128 = torch.nn.functional.silu(batch_norm_136, inplace=True)
        batch_norm_136 = None
        input_33 = silu_126 + silu_128
        silu_126 = silu_128 = None
        conv2d_137 = torch.conv2d(
            input_33,
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
        batch_norm_137 = torch.nn.functional.batch_norm(
            conv2d_137,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_137 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_129 = torch.nn.functional.silu(batch_norm_137, inplace=True)
        batch_norm_137 = None
        conv2d_138 = torch.conv2d(
            silu_129,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_129 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_138 = torch.nn.functional.batch_norm(
            conv2d_138,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_138 = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_130 = torch.nn.functional.silu(batch_norm_138, inplace=True)
        batch_norm_138 = None
        input_34 = input_33 + silu_130
        input_33 = silu_130 = None
        conv2d_139 = torch.conv2d(
            getitem_23,
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
        batch_norm_139 = torch.nn.functional.batch_norm(
            conv2d_139,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_139 = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_131 = torch.nn.functional.silu(batch_norm_139, inplace=True)
        batch_norm_139 = None
        cat_27 = torch.cat((input_34, silu_131), 1)
        input_34 = silu_131 = None
        conv2d_140 = torch.conv2d(
            cat_27,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_27 = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_140 = torch.nn.functional.batch_norm(
            conv2d_140,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_140 = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_132 = torch.nn.functional.silu(batch_norm_140, inplace=True)
        batch_norm_140 = None
        conv2d_141 = torch.conv2d(
            getitem_23,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_141 = torch.nn.functional.batch_norm(
            conv2d_141,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_141 = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_133 = torch.nn.functional.silu(batch_norm_141, inplace=True)
        batch_norm_141 = None
        conv2d_142 = torch.conv2d(
            silu_133,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_142 = torch.nn.functional.batch_norm(
            conv2d_142,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_142 = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_134 = torch.nn.functional.silu(batch_norm_142, inplace=True)
        batch_norm_142 = None
        conv2d_143 = torch.conv2d(
            silu_134,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_134 = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_143 = torch.nn.functional.batch_norm(
            conv2d_143,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_143 = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_135 = torch.nn.functional.silu(batch_norm_143, inplace=True)
        batch_norm_143 = None
        input_35 = silu_133 + silu_135
        silu_133 = silu_135 = None
        conv2d_144 = torch.conv2d(
            input_35,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_144 = torch.nn.functional.batch_norm(
            conv2d_144,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_144 = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_136 = torch.nn.functional.silu(batch_norm_144, inplace=True)
        batch_norm_144 = None
        conv2d_145 = torch.conv2d(
            silu_136,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_136 = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_145 = torch.nn.functional.batch_norm(
            conv2d_145,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_145 = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_137 = torch.nn.functional.silu(batch_norm_145, inplace=True)
        batch_norm_145 = None
        input_36 = input_35 + silu_137
        input_35 = silu_137 = None
        conv2d_146 = torch.conv2d(
            getitem_23,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_146 = torch.nn.functional.batch_norm(
            conv2d_146,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_146 = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_138 = torch.nn.functional.silu(batch_norm_146, inplace=True)
        batch_norm_146 = None
        cat_28 = torch.cat((input_36, silu_138), 1)
        input_36 = silu_138 = None
        conv2d_147 = torch.conv2d(
            cat_28,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_28 = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_147 = torch.nn.functional.batch_norm(
            conv2d_147,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_147 = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_22_modules_m_modules_1_modules_cv3_modules_bn_parameters_bias_ = (None)
        silu_139 = torch.nn.functional.silu(batch_norm_147, inplace=True)
        batch_norm_147 = None
        cat_29 = torch.cat([getitem_22, getitem_23, silu_132, silu_139], 1)
        getitem_22 = getitem_23 = silu_132 = silu_139 = None
        conv2d_148 = torch.conv2d(
            cat_29,
            l_self_modules_model_modules_22_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_29 = (
            l_self_modules_model_modules_22_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_148 = torch.nn.functional.batch_norm(
            conv2d_148,
            l_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_148 = (
            l_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_22_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_22_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_30 = torch.nn.functional.silu(batch_norm_148, inplace=True)
        batch_norm_148 = None
        conv2d_149 = torch.conv2d(
            x_24,
            l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_149 = torch.nn.functional.batch_norm(
            conv2d_149,
            l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_149 = l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_proto_modules_cv1_modules_bn_parameters_bias_ = (None)
        silu_141 = torch.nn.functional.silu(batch_norm_149, inplace=True)
        batch_norm_149 = None
        conv_transpose2d = torch.conv_transpose2d(
            silu_141,
            l_self_modules_model_modules_23_modules_proto_modules_upsample_parameters_weight_,
            l_self_modules_model_modules_23_modules_proto_modules_upsample_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        silu_141 = l_self_modules_model_modules_23_modules_proto_modules_upsample_parameters_weight_ = l_self_modules_model_modules_23_modules_proto_modules_upsample_parameters_bias_ = (None)
        conv2d_150 = torch.conv2d(
            conv_transpose2d,
            l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        conv_transpose2d = l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_conv_parameters_weight_ = (None)
        batch_norm_150 = torch.nn.functional.batch_norm(
            conv2d_150,
            l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_150 = l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_proto_modules_cv2_modules_bn_parameters_bias_ = (None)
        silu_142 = torch.nn.functional.silu(batch_norm_150, inplace=True)
        batch_norm_150 = None
        conv2d_151 = torch.conv2d(
            silu_142,
            l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        silu_142 = l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_conv_parameters_weight_ = (None)
        batch_norm_151 = torch.nn.functional.batch_norm(
            conv2d_151,
            l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_151 = l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_proto_modules_cv3_modules_bn_parameters_bias_ = (None)
        p = torch.nn.functional.silu(batch_norm_151, inplace=True)
        batch_norm_151 = None
        conv2d_152 = torch.conv2d(
            x_24,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_152 = torch.nn.functional.batch_norm(
            conv2d_152,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_152 = l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv4_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_37 = torch.nn.functional.silu(batch_norm_152, inplace=True)
        batch_norm_152 = None
        conv2d_153 = torch.conv2d(
            input_37,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_37 = l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_153 = torch.nn.functional.batch_norm(
            conv2d_153,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_153 = l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv4_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_38 = torch.nn.functional.silu(batch_norm_153, inplace=True)
        batch_norm_153 = None
        input_39 = torch.conv2d(
            input_38,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv4_modules_0_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_38 = l_self_modules_model_modules_23_modules_cv4_modules_0_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv4_modules_0_modules_2_parameters_bias_ = (None)
        view_4 = input_39.view(1, 32, -1)
        input_39 = None
        conv2d_155 = torch.conv2d(
            x_27,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_154 = torch.nn.functional.batch_norm(
            conv2d_155,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_155 = l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv4_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_40 = torch.nn.functional.silu(batch_norm_154, inplace=True)
        batch_norm_154 = None
        conv2d_156 = torch.conv2d(
            input_40,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_40 = l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_155 = torch.nn.functional.batch_norm(
            conv2d_156,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_156 = l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv4_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_41 = torch.nn.functional.silu(batch_norm_155, inplace=True)
        batch_norm_155 = None
        input_42 = torch.conv2d(
            input_41,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv4_modules_1_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_41 = l_self_modules_model_modules_23_modules_cv4_modules_1_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv4_modules_1_modules_2_parameters_bias_ = (None)
        view_5 = input_42.view(1, 32, -1)
        input_42 = None
        conv2d_158 = torch.conv2d(
            x_30,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_156 = torch.nn.functional.batch_norm(
            conv2d_158,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_158 = l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv4_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        input_43 = torch.nn.functional.silu(batch_norm_156, inplace=True)
        batch_norm_156 = None
        conv2d_159 = torch.conv2d(
            input_43,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_43 = l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_157 = torch.nn.functional.batch_norm(
            conv2d_159,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_159 = l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv4_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        input_44 = torch.nn.functional.silu(batch_norm_157, inplace=True)
        batch_norm_157 = None
        input_45 = torch.conv2d(
            input_44,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv4_modules_2_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_44 = l_self_modules_model_modules_23_modules_cv4_modules_2_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv4_modules_2_modules_2_parameters_bias_ = (None)
        view_6 = input_45.view(1, 32, -1)
        input_45 = None
        mc = torch.cat([view_4, view_5, view_6], 2)
        view_4 = view_5 = view_6 = None
        conv2d_161 = torch.conv2d(
            x_24,
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
        batch_norm_158 = torch.nn.functional.batch_norm(
            conv2d_161,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_161 = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_46 = torch.nn.functional.silu(batch_norm_158, inplace=True)
        batch_norm_158 = None
        conv2d_162 = torch.conv2d(
            input_46,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_46 = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_159 = torch.nn.functional.batch_norm(
            conv2d_162,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_162 = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_47 = torch.nn.functional.silu(batch_norm_159, inplace=True)
        batch_norm_159 = None
        input_48 = torch.conv2d(
            input_47,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_47 = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_0_modules_2_parameters_bias_ = (None)
        conv2d_164 = torch.conv2d(
            x_24,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        x_24 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_160 = torch.nn.functional.batch_norm(
            conv2d_164,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_164 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_49 = torch.nn.functional.silu(batch_norm_160, inplace=True)
        batch_norm_160 = None
        conv2d_165 = torch.conv2d(
            input_49,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_49 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_161 = torch.nn.functional.batch_norm(
            conv2d_165,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_165 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_50 = torch.nn.functional.silu(batch_norm_161, inplace=True)
        batch_norm_161 = None
        conv2d_166 = torch.conv2d(
            input_50,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        input_50 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_162 = torch.nn.functional.batch_norm(
            conv2d_166,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_166 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_51 = torch.nn.functional.silu(batch_norm_162, inplace=True)
        batch_norm_162 = None
        conv2d_167 = torch.conv2d(
            input_51,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_51 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_163 = torch.nn.functional.batch_norm(
            conv2d_167,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_167 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_52 = torch.nn.functional.silu(batch_norm_163, inplace=True)
        batch_norm_163 = None
        input_53 = torch.conv2d(
            input_52,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_52 = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_0_modules_2_parameters_bias_ = (None)
        xi = torch.cat((input_48, input_53), 1)
        input_48 = input_53 = None
        conv2d_169 = torch.conv2d(
            x_27,
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
        batch_norm_164 = torch.nn.functional.batch_norm(
            conv2d_169,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_169 = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_54 = torch.nn.functional.silu(batch_norm_164, inplace=True)
        batch_norm_164 = None
        conv2d_170 = torch.conv2d(
            input_54,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_54 = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_165 = torch.nn.functional.batch_norm(
            conv2d_170,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_170 = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_55 = torch.nn.functional.silu(batch_norm_165, inplace=True)
        batch_norm_165 = None
        input_56 = torch.conv2d(
            input_55,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_55 = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_1_modules_2_parameters_bias_ = (None)
        conv2d_172 = torch.conv2d(
            x_27,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_27 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_166 = torch.nn.functional.batch_norm(
            conv2d_172,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_172 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_57 = torch.nn.functional.silu(batch_norm_166, inplace=True)
        batch_norm_166 = None
        conv2d_173 = torch.conv2d(
            input_57,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_57 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_167 = torch.nn.functional.batch_norm(
            conv2d_173,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_173 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_58 = torch.nn.functional.silu(batch_norm_167, inplace=True)
        batch_norm_167 = None
        conv2d_174 = torch.conv2d(
            input_58,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        input_58 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_168 = torch.nn.functional.batch_norm(
            conv2d_174,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_174 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_59 = torch.nn.functional.silu(batch_norm_168, inplace=True)
        batch_norm_168 = None
        conv2d_175 = torch.conv2d(
            input_59,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_59 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_169 = torch.nn.functional.batch_norm(
            conv2d_175,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_175 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_60 = torch.nn.functional.silu(batch_norm_169, inplace=True)
        batch_norm_169 = None
        input_61 = torch.conv2d(
            input_60,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_60 = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_1_modules_2_parameters_bias_ = (None)
        xi_1 = torch.cat((input_56, input_61), 1)
        input_56 = input_61 = None
        conv2d_177 = torch.conv2d(
            x_30,
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
        batch_norm_170 = torch.nn.functional.batch_norm(
            conv2d_177,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_177 = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        input_62 = torch.nn.functional.silu(batch_norm_170, inplace=True)
        batch_norm_170 = None
        conv2d_178 = torch.conv2d(
            input_62,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_62 = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_171 = torch.nn.functional.batch_norm(
            conv2d_178,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_178 = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        input_63 = torch.nn.functional.silu(batch_norm_171, inplace=True)
        batch_norm_171 = None
        input_64 = torch.conv2d(
            input_63,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_63 = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv2_modules_2_modules_2_parameters_bias_ = (None)
        conv2d_180 = torch.conv2d(
            x_30,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_30 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_172 = torch.nn.functional.batch_norm(
            conv2d_180,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_180 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_65 = torch.nn.functional.silu(batch_norm_172, inplace=True)
        batch_norm_172 = None
        conv2d_181 = torch.conv2d(
            input_65,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_65 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_173 = torch.nn.functional.batch_norm(
            conv2d_181,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_181 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_66 = torch.nn.functional.silu(batch_norm_173, inplace=True)
        batch_norm_173 = None
        conv2d_182 = torch.conv2d(
            input_66,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        input_66 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_174 = torch.nn.functional.batch_norm(
            conv2d_182,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_182 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_67 = torch.nn.functional.silu(batch_norm_174, inplace=True)
        batch_norm_174 = None
        conv2d_183 = torch.conv2d(
            input_67,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_67 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_175 = torch.nn.functional.batch_norm(
            conv2d_183,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_183 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_68 = torch.nn.functional.silu(batch_norm_175, inplace=True)
        batch_norm_175 = None
        input_69 = torch.conv2d(
            input_68,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_weight_,
            l_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_68 = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_weight_ = l_self_modules_model_modules_23_modules_cv3_modules_2_modules_2_parameters_bias_ = (None)
        xi_2 = torch.cat((input_64, input_69), 1)
        input_64 = input_69 = None
        view_7 = xi.view(1, 144, -1)
        view_8 = xi_1.view(1, 144, -1)
        view_9 = xi_2.view(1, 144, -1)
        x_cat = torch.cat([view_7, view_8, view_9], 2)
        view_7 = view_8 = view_9 = None
        x_31 = l_self_modules_model_modules_23_stride[0]
        x_32 = l_self_modules_model_modules_23_stride[1]
        x_33 = l_self_modules_model_modules_23_stride[2]
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
        view_10 = stack.view(-1, 2)
        stack = None
        _local_scalar_dense = torch.ops.aten._local_scalar_dense(x_31)
        x_31 = None
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
        view_11 = stack_1.view(-1, 2)
        stack_1 = None
        _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense(x_32)
        x_32 = None
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
        view_12 = stack_2.view(-1, 2)
        stack_2 = None
        _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense(x_33)
        x_33 = None
        full_2 = torch.full(
            (400, 1),
            _local_scalar_dense_2,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        _local_scalar_dense_2 = None
        x_34 = torch.cat([view_10, view_11, view_12])
        view_10 = view_11 = view_12 = None
        x_35 = torch.cat([full, full_1, full_2])
        full = full_1 = full_2 = None
        transpose_4 = x_34.transpose(0, 1)
        x_34 = None
        transpose_5 = x_35.transpose(0, 1)
        x_35 = None
        split_3 = x_cat.split((64, 80), 1)
        x_cat = None
        box = split_3[0]
        cls = split_3[1]
        split_3 = None
        view_13 = box.view(1, 4, 16, 8400)
        box = None
        transpose_6 = view_13.transpose(2, 1)
        view_13 = None
        softmax_2 = transpose_6.softmax(1)
        transpose_6 = None
        conv2d_185 = torch.conv2d(
            softmax_2,
            l_self_modules_model_modules_23_modules_dfl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        softmax_2 = (
            l_self_modules_model_modules_23_modules_dfl_modules_conv_parameters_weight_
        ) = None
        view_14 = conv2d_185.view(1, 4, 8400)
        conv2d_185 = None
        unsqueeze = transpose_4.unsqueeze(0)
        chunk_8 = view_14.chunk(2, 1)
        view_14 = None
        lt = chunk_8[0]
        rb = chunk_8[1]
        chunk_8 = None
        x1y1 = unsqueeze - lt
        lt = None
        x2y2 = unsqueeze + rb
        unsqueeze = rb = None
        add_45 = x1y1 + x2y2
        c_xy = add_45 / 2
        add_45 = None
        wh = x2y2 - x1y1
        x2y2 = x1y1 = None
        cat_37 = torch.cat((c_xy, wh), 1)
        c_xy = wh = None
        dbox = cat_37 * transpose_5
        cat_37 = None
        sigmoid = cls.sigmoid()
        cls = None
        y = torch.cat((dbox, sigmoid), 1)
        dbox = sigmoid = None
        cat_39 = torch.cat([y, mc], 1)
        y = None
        return (cat_39, xi, xi_1, xi_2, mc, p, transpose_5, transpose_4)
