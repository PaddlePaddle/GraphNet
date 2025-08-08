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
        L_self_modules_model_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_6_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_6_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_6_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_7_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_7_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_7_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_6_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_6_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_6_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_7_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_7_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_7_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_8_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_8_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_8_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_9_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_9_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_9_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_10_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_10_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_10_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_11_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_11_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_11_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_7_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_7_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_model_modules_10_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_11_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_11_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_14_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_14_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_14_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_14_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_14_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_14_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_18_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_18_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_18_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_20_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_20_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_22_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_22_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_22_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_23_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_23_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_24_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_24_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_24_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_24_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_24_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_26_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_26_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_26_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_26_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_26_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_27_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_27_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_27_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_27_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_27_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_27_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_27_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_28_stride: torch.Tensor,
        L_self_modules_model_modules_28_modules_dfl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_model_modules_2_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_2_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_2_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_2_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_2_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_2_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_2_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_2_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_2_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_2_modules_3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_3_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_2_modules_3_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_3_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_3_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_2_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_4_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_3_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_3_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_3_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_3_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_3_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_5_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_5_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_5_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_5_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_5_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_5_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_6_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_6_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_6_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_6_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_6_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_6_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_7_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_7_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_7_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_7_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_7_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_7_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_6_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_3_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_3_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_3_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_3_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_3_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_5_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_5_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_5_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_5_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_5_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_5_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_6_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_6_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_6_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_6_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_6_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_6_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_7_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_7_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_7_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_7_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_7_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_7_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_8_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_8_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_8_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_8_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_8_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_8_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_9_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_9_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_9_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_9_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_9_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_9_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_10_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_10_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_10_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_10_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_10_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_10_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_11_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_11_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_11_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_11_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_11_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_11_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_8_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_8_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_8_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_8_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_8_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_8_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_8_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_8_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_8_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_8_modules_3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_3_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_8_modules_3_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_3_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_3_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_8_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_10_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_10_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_10_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_10_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_10_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_10_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_10_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_10_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_10_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_10_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_11_parameters_weight_ = (
            L_self_modules_model_modules_11_parameters_weight_
        )
        l_self_modules_model_modules_11_parameters_bias_ = (
            L_self_modules_model_modules_11_parameters_bias_
        )
        l_self_modules_model_modules_13_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_13_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_13_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_13_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_13_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_13_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_13_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_13_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_13_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_13_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_14_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_14_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_14_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_14_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_14_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_14_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_14_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_14_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_14_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_14_modules_3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_3_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_14_modules_3_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_3_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_3_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_14_modules_3_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_14_modules_4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_14_modules_4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_14_modules_4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_14_modules_5_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_5_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_14_modules_5_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_5_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_5_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_14_modules_5_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_15_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_15_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_15_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_15_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_15_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_15_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_15_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_15_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_15_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_15_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_16_parameters_weight_ = (
            L_self_modules_model_modules_16_parameters_weight_
        )
        l_self_modules_model_modules_16_parameters_bias_ = (
            L_self_modules_model_modules_16_parameters_bias_
        )
        l_self_modules_model_modules_18_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_18_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_18_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_18_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_18_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_18_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_18_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_18_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_18_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_18_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_19_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_19_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_19_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_19_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_19_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_19_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_19_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_19_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_19_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_19_modules_3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_3_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_19_modules_3_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_3_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_3_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_19_modules_3_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_19_modules_4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_19_modules_4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_19_modules_4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_19_modules_5_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_5_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_19_modules_5_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_5_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_5_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_19_modules_5_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_22_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_22_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_22_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_22_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_22_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_22_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_22_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_22_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_22_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_22_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_23_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_23_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_23_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_23_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_23_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_23_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_23_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_23_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_23_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_23_modules_3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_3_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_23_modules_3_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_3_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_3_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_23_modules_3_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_23_modules_4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_23_modules_4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_23_modules_4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_23_modules_5_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_5_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_23_modules_5_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_23_modules_5_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_23_modules_5_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_23_modules_5_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_24_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_24_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_24_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_24_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_24_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_24_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_24_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_24_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_24_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_24_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_26_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_26_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_26_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_26_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_26_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_26_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_26_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_26_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_26_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_26_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_27_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_27_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_27_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_27_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_27_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_27_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_27_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_27_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_27_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_27_modules_3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_3_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_27_modules_3_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_3_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_3_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_27_modules_3_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_27_modules_4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_27_modules_4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_27_modules_4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_27_modules_5_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_5_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_27_modules_5_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_27_modules_5_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_27_modules_5_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_27_modules_5_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_bias_ = L_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_bias_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_bias_ = L_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_bias_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_bias_ = L_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_bias_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_bias_ = L_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_bias_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_weight_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_weight_
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_bias_ = L_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_bias_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_weight_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_weight_
        l_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_bias_ = L_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_bias_
        l_self_modules_model_modules_28_stride = L_self_modules_model_modules_28_stride
        l_self_modules_model_modules_28_modules_dfl_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_28_modules_dfl_modules_conv_parameters_weight_
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
        x = torch.nn.functional.relu(batch_norm, inplace=True)
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
        x_1 = torch.nn.functional.relu(batch_norm_1, inplace=True)
        batch_norm_1 = None
        conv2d_2 = torch.conv2d(
            x_1,
            l_self_modules_model_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_1 = (
            l_self_modules_model_modules_2_modules_0_modules_conv_parameters_weight_
        ) = None
        batch_norm_2 = torch.nn.functional.batch_norm(
            conv2d_2,
            l_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_2 = (
            l_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_2_modules_0_modules_bn_parameters_bias_ = None
        input_1 = torch.nn.functional.relu(batch_norm_2, inplace=True)
        batch_norm_2 = None
        conv2d_3 = torch.conv2d(
            input_1,
            l_self_modules_model_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_1 = (
            l_self_modules_model_modules_2_modules_1_modules_conv_parameters_weight_
        ) = None
        batch_norm_3 = torch.nn.functional.batch_norm(
            conv2d_3,
            l_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_3 = (
            l_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_2_modules_1_modules_bn_parameters_bias_ = None
        input_2 = torch.nn.functional.relu(batch_norm_3, inplace=True)
        batch_norm_3 = None
        conv2d_4 = torch.conv2d(
            input_2,
            l_self_modules_model_modules_2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_2 = (
            l_self_modules_model_modules_2_modules_2_modules_conv_parameters_weight_
        ) = None
        batch_norm_4 = torch.nn.functional.batch_norm(
            conv2d_4,
            l_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_4 = (
            l_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_2_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_2_modules_2_modules_bn_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(batch_norm_4, inplace=True)
        batch_norm_4 = None
        conv2d_5 = torch.conv2d(
            input_3,
            l_self_modules_model_modules_2_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = (
            l_self_modules_model_modules_2_modules_3_modules_conv_parameters_weight_
        ) = None
        batch_norm_5 = torch.nn.functional.batch_norm(
            conv2d_5,
            l_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_5 = (
            l_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_2_modules_3_modules_bn_parameters_bias_ = None
        input_4 = torch.nn.functional.relu(batch_norm_5, inplace=True)
        batch_norm_5 = None
        conv2d_6 = torch.conv2d(
            input_4,
            l_self_modules_model_modules_3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_model_modules_3_modules_conv_parameters_weight_ = None
        batch_norm_6 = torch.nn.functional.batch_norm(
            conv2d_6,
            l_self_modules_model_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_6 = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_3_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(batch_norm_6, inplace=True)
        batch_norm_6 = None
        conv2d_7 = torch.conv2d(
            x_2,
            l_self_modules_model_modules_4_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = (
            l_self_modules_model_modules_4_modules_0_modules_conv_parameters_weight_
        ) = None
        batch_norm_7 = torch.nn.functional.batch_norm(
            conv2d_7,
            l_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_7 = (
            l_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_4_modules_0_modules_bn_parameters_bias_ = None
        input_5 = torch.nn.functional.relu(batch_norm_7, inplace=True)
        batch_norm_7 = None
        conv2d_8 = torch.conv2d(
            input_5,
            l_self_modules_model_modules_4_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_5 = (
            l_self_modules_model_modules_4_modules_1_modules_conv_parameters_weight_
        ) = None
        batch_norm_8 = torch.nn.functional.batch_norm(
            conv2d_8,
            l_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_8 = (
            l_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_4_modules_1_modules_bn_parameters_bias_ = None
        input_6 = torch.nn.functional.relu(batch_norm_8, inplace=True)
        batch_norm_8 = None
        conv2d_9 = torch.conv2d(
            input_6,
            l_self_modules_model_modules_4_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_6 = (
            l_self_modules_model_modules_4_modules_2_modules_conv_parameters_weight_
        ) = None
        batch_norm_9 = torch.nn.functional.batch_norm(
            conv2d_9,
            l_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_9 = (
            l_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_2_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_4_modules_2_modules_bn_parameters_bias_ = None
        input_7 = torch.nn.functional.relu(batch_norm_9, inplace=True)
        batch_norm_9 = None
        conv2d_10 = torch.conv2d(
            input_7,
            l_self_modules_model_modules_4_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_7 = (
            l_self_modules_model_modules_4_modules_3_modules_conv_parameters_weight_
        ) = None
        batch_norm_10 = torch.nn.functional.batch_norm(
            conv2d_10,
            l_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_10 = (
            l_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_4_modules_3_modules_bn_parameters_bias_ = None
        input_8 = torch.nn.functional.relu(batch_norm_10, inplace=True)
        batch_norm_10 = None
        conv2d_11 = torch.conv2d(
            input_8,
            l_self_modules_model_modules_4_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_8 = (
            l_self_modules_model_modules_4_modules_4_modules_conv_parameters_weight_
        ) = None
        batch_norm_11 = torch.nn.functional.batch_norm(
            conv2d_11,
            l_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_11 = (
            l_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_4_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_4_modules_4_modules_bn_parameters_bias_ = None
        input_9 = torch.nn.functional.relu(batch_norm_11, inplace=True)
        batch_norm_11 = None
        conv2d_12 = torch.conv2d(
            input_9,
            l_self_modules_model_modules_4_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_9 = (
            l_self_modules_model_modules_4_modules_5_modules_conv_parameters_weight_
        ) = None
        batch_norm_12 = torch.nn.functional.batch_norm(
            conv2d_12,
            l_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_5_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_12 = (
            l_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_5_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_4_modules_5_modules_bn_parameters_bias_ = None
        input_10 = torch.nn.functional.relu(batch_norm_12, inplace=True)
        batch_norm_12 = None
        conv2d_13 = torch.conv2d(
            input_10,
            l_self_modules_model_modules_4_modules_6_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_10 = (
            l_self_modules_model_modules_4_modules_6_modules_conv_parameters_weight_
        ) = None
        batch_norm_13 = torch.nn.functional.batch_norm(
            conv2d_13,
            l_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_6_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_6_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_13 = (
            l_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_6_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_4_modules_6_modules_bn_parameters_bias_ = None
        input_11 = torch.nn.functional.relu(batch_norm_13, inplace=True)
        batch_norm_13 = None
        conv2d_14 = torch.conv2d(
            input_11,
            l_self_modules_model_modules_4_modules_7_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_11 = (
            l_self_modules_model_modules_4_modules_7_modules_conv_parameters_weight_
        ) = None
        batch_norm_14 = torch.nn.functional.batch_norm(
            conv2d_14,
            l_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_7_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_7_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_14 = (
            l_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_7_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_4_modules_7_modules_bn_parameters_bias_ = None
        input_12 = torch.nn.functional.relu(batch_norm_14, inplace=True)
        batch_norm_14 = None
        conv2d_15 = torch.conv2d(
            input_12,
            l_self_modules_model_modules_5_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_5_modules_conv_parameters_weight_ = None
        batch_norm_15 = torch.nn.functional.batch_norm(
            conv2d_15,
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_5_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_15 = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_5_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_5_modules_bn_parameters_bias_ = None
        x_3 = torch.nn.functional.relu(batch_norm_15, inplace=True)
        batch_norm_15 = None
        conv2d_16 = torch.conv2d(
            x_3,
            l_self_modules_model_modules_6_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_3 = (
            l_self_modules_model_modules_6_modules_0_modules_conv_parameters_weight_
        ) = None
        batch_norm_16 = torch.nn.functional.batch_norm(
            conv2d_16,
            l_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_16 = (
            l_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_0_modules_bn_parameters_bias_ = None
        input_13 = torch.nn.functional.relu(batch_norm_16, inplace=True)
        batch_norm_16 = None
        conv2d_17 = torch.conv2d(
            input_13,
            l_self_modules_model_modules_6_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_13 = (
            l_self_modules_model_modules_6_modules_1_modules_conv_parameters_weight_
        ) = None
        batch_norm_17 = torch.nn.functional.batch_norm(
            conv2d_17,
            l_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_17 = (
            l_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_1_modules_bn_parameters_bias_ = None
        input_14 = torch.nn.functional.relu(batch_norm_17, inplace=True)
        batch_norm_17 = None
        conv2d_18 = torch.conv2d(
            input_14,
            l_self_modules_model_modules_6_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_14 = (
            l_self_modules_model_modules_6_modules_2_modules_conv_parameters_weight_
        ) = None
        batch_norm_18 = torch.nn.functional.batch_norm(
            conv2d_18,
            l_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_18 = (
            l_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_2_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_2_modules_bn_parameters_bias_ = None
        input_15 = torch.nn.functional.relu(batch_norm_18, inplace=True)
        batch_norm_18 = None
        conv2d_19 = torch.conv2d(
            input_15,
            l_self_modules_model_modules_6_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_15 = (
            l_self_modules_model_modules_6_modules_3_modules_conv_parameters_weight_
        ) = None
        batch_norm_19 = torch.nn.functional.batch_norm(
            conv2d_19,
            l_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_19 = (
            l_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_3_modules_bn_parameters_bias_ = None
        input_16 = torch.nn.functional.relu(batch_norm_19, inplace=True)
        batch_norm_19 = None
        conv2d_20 = torch.conv2d(
            input_16,
            l_self_modules_model_modules_6_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_16 = (
            l_self_modules_model_modules_6_modules_4_modules_conv_parameters_weight_
        ) = None
        batch_norm_20 = torch.nn.functional.batch_norm(
            conv2d_20,
            l_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_20 = (
            l_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_4_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_4_modules_bn_parameters_bias_ = None
        input_17 = torch.nn.functional.relu(batch_norm_20, inplace=True)
        batch_norm_20 = None
        conv2d_21 = torch.conv2d(
            input_17,
            l_self_modules_model_modules_6_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_17 = (
            l_self_modules_model_modules_6_modules_5_modules_conv_parameters_weight_
        ) = None
        batch_norm_21 = torch.nn.functional.batch_norm(
            conv2d_21,
            l_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_5_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_21 = (
            l_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_5_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_5_modules_bn_parameters_bias_ = None
        input_18 = torch.nn.functional.relu(batch_norm_21, inplace=True)
        batch_norm_21 = None
        conv2d_22 = torch.conv2d(
            input_18,
            l_self_modules_model_modules_6_modules_6_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_18 = (
            l_self_modules_model_modules_6_modules_6_modules_conv_parameters_weight_
        ) = None
        batch_norm_22 = torch.nn.functional.batch_norm(
            conv2d_22,
            l_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_6_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_6_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_22 = (
            l_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_6_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_6_modules_bn_parameters_bias_ = None
        input_19 = torch.nn.functional.relu(batch_norm_22, inplace=True)
        batch_norm_22 = None
        conv2d_23 = torch.conv2d(
            input_19,
            l_self_modules_model_modules_6_modules_7_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_19 = (
            l_self_modules_model_modules_6_modules_7_modules_conv_parameters_weight_
        ) = None
        batch_norm_23 = torch.nn.functional.batch_norm(
            conv2d_23,
            l_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_7_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_7_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_23 = (
            l_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_7_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_7_modules_bn_parameters_bias_ = None
        input_20 = torch.nn.functional.relu(batch_norm_23, inplace=True)
        batch_norm_23 = None
        conv2d_24 = torch.conv2d(
            input_20,
            l_self_modules_model_modules_6_modules_8_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_20 = (
            l_self_modules_model_modules_6_modules_8_modules_conv_parameters_weight_
        ) = None
        batch_norm_24 = torch.nn.functional.batch_norm(
            conv2d_24,
            l_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_8_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_8_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_24 = (
            l_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_8_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_8_modules_bn_parameters_bias_ = None
        input_21 = torch.nn.functional.relu(batch_norm_24, inplace=True)
        batch_norm_24 = None
        conv2d_25 = torch.conv2d(
            input_21,
            l_self_modules_model_modules_6_modules_9_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_21 = (
            l_self_modules_model_modules_6_modules_9_modules_conv_parameters_weight_
        ) = None
        batch_norm_25 = torch.nn.functional.batch_norm(
            conv2d_25,
            l_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_9_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_9_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_25 = (
            l_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_9_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_9_modules_bn_parameters_bias_ = None
        input_22 = torch.nn.functional.relu(batch_norm_25, inplace=True)
        batch_norm_25 = None
        conv2d_26 = torch.conv2d(
            input_22,
            l_self_modules_model_modules_6_modules_10_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_22 = (
            l_self_modules_model_modules_6_modules_10_modules_conv_parameters_weight_
        ) = None
        batch_norm_26 = torch.nn.functional.batch_norm(
            conv2d_26,
            l_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_10_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_10_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_26 = (
            l_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_10_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_10_modules_bn_parameters_bias_ = None
        input_23 = torch.nn.functional.relu(batch_norm_26, inplace=True)
        batch_norm_26 = None
        conv2d_27 = torch.conv2d(
            input_23,
            l_self_modules_model_modules_6_modules_11_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_23 = (
            l_self_modules_model_modules_6_modules_11_modules_conv_parameters_weight_
        ) = None
        batch_norm_27 = torch.nn.functional.batch_norm(
            conv2d_27,
            l_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_11_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_11_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_27 = (
            l_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_11_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_11_modules_bn_parameters_bias_ = None
        input_24 = torch.nn.functional.relu(batch_norm_27, inplace=True)
        batch_norm_27 = None
        conv2d_28 = torch.conv2d(
            input_24,
            l_self_modules_model_modules_7_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_7_modules_conv_parameters_weight_ = None
        batch_norm_28 = torch.nn.functional.batch_norm(
            conv2d_28,
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_7_modules_bn_parameters_weight_,
            l_self_modules_model_modules_7_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_28 = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_7_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_7_modules_bn_parameters_bias_ = None
        x_4 = torch.nn.functional.relu(batch_norm_28, inplace=True)
        batch_norm_28 = None
        conv2d_29 = torch.conv2d(
            x_4,
            l_self_modules_model_modules_8_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_4 = (
            l_self_modules_model_modules_8_modules_0_modules_conv_parameters_weight_
        ) = None
        batch_norm_29 = torch.nn.functional.batch_norm(
            conv2d_29,
            l_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_29 = (
            l_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_8_modules_0_modules_bn_parameters_bias_ = None
        input_25 = torch.nn.functional.relu(batch_norm_29, inplace=True)
        batch_norm_29 = None
        conv2d_30 = torch.conv2d(
            input_25,
            l_self_modules_model_modules_8_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_25 = (
            l_self_modules_model_modules_8_modules_1_modules_conv_parameters_weight_
        ) = None
        batch_norm_30 = torch.nn.functional.batch_norm(
            conv2d_30,
            l_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_30 = (
            l_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_8_modules_1_modules_bn_parameters_bias_ = None
        input_26 = torch.nn.functional.relu(batch_norm_30, inplace=True)
        batch_norm_30 = None
        conv2d_31 = torch.conv2d(
            input_26,
            l_self_modules_model_modules_8_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_26 = (
            l_self_modules_model_modules_8_modules_2_modules_conv_parameters_weight_
        ) = None
        batch_norm_31 = torch.nn.functional.batch_norm(
            conv2d_31,
            l_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_31 = (
            l_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_2_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_8_modules_2_modules_bn_parameters_bias_ = None
        input_27 = torch.nn.functional.relu(batch_norm_31, inplace=True)
        batch_norm_31 = None
        conv2d_32 = torch.conv2d(
            input_27,
            l_self_modules_model_modules_8_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_27 = (
            l_self_modules_model_modules_8_modules_3_modules_conv_parameters_weight_
        ) = None
        batch_norm_32 = torch.nn.functional.batch_norm(
            conv2d_32,
            l_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_32 = (
            l_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_8_modules_3_modules_bn_parameters_bias_ = None
        input_28 = torch.nn.functional.relu(batch_norm_32, inplace=True)
        batch_norm_32 = None
        conv2d_33 = torch.conv2d(
            input_28,
            l_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_28 = (
            l_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_
        ) = None
        batch_norm_33 = torch.nn.functional.batch_norm(
            conv2d_33,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_33 = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_
        ) = None
        relu_33 = torch.nn.functional.relu(batch_norm_33, inplace=True)
        batch_norm_33 = None
        max_pool2d = torch.nn.functional.max_pool2d(
            relu_33, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_1 = torch.nn.functional.max_pool2d(
            relu_33, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_2 = torch.nn.functional.max_pool2d(
            relu_33, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        cat = torch.cat([relu_33, max_pool2d, max_pool2d_1, max_pool2d_2], 1)
        relu_33 = max_pool2d = max_pool2d_1 = max_pool2d_2 = None
        conv2d_34 = torch.conv2d(
            cat,
            l_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = (
            l_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_
        ) = None
        batch_norm_34 = torch.nn.functional.batch_norm(
            conv2d_34,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_34 = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_
        ) = None
        x_5 = torch.nn.functional.relu(batch_norm_34, inplace=True)
        batch_norm_34 = None
        conv2d_35 = torch.conv2d(
            x_5,
            l_self_modules_model_modules_10_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_model_modules_10_modules_conv_parameters_weight_ = None
        batch_norm_35 = torch.nn.functional.batch_norm(
            conv2d_35,
            l_self_modules_model_modules_10_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_35 = (
            l_self_modules_model_modules_10_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_10_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_10_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_10_modules_bn_parameters_bias_ = None
        x_6 = torch.nn.functional.relu(batch_norm_35, inplace=True)
        batch_norm_35 = None
        x_7 = torch.conv_transpose2d(
            x_6,
            l_self_modules_model_modules_11_parameters_weight_,
            l_self_modules_model_modules_11_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        l_self_modules_model_modules_11_parameters_weight_ = (
            l_self_modules_model_modules_11_parameters_bias_
        ) = None
        x_8 = torch.cat([x_7, input_24], 1)
        x_7 = input_24 = None
        conv2d_36 = torch.conv2d(
            x_8,
            l_self_modules_model_modules_13_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_model_modules_13_modules_conv_parameters_weight_ = None
        batch_norm_36 = torch.nn.functional.batch_norm(
            conv2d_36,
            l_self_modules_model_modules_13_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_36 = (
            l_self_modules_model_modules_13_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_13_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_13_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_13_modules_bn_parameters_bias_ = None
        x_9 = torch.nn.functional.relu(batch_norm_36, inplace=True)
        batch_norm_36 = None
        conv2d_37 = torch.conv2d(
            x_9,
            l_self_modules_model_modules_14_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_9 = (
            l_self_modules_model_modules_14_modules_0_modules_conv_parameters_weight_
        ) = None
        batch_norm_37 = torch.nn.functional.batch_norm(
            conv2d_37,
            l_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_14_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_14_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_37 = (
            l_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_14_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_14_modules_0_modules_bn_parameters_bias_ = None
        input_29 = torch.nn.functional.relu(batch_norm_37, inplace=True)
        batch_norm_37 = None
        conv2d_38 = torch.conv2d(
            input_29,
            l_self_modules_model_modules_14_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_29 = (
            l_self_modules_model_modules_14_modules_1_modules_conv_parameters_weight_
        ) = None
        batch_norm_38 = torch.nn.functional.batch_norm(
            conv2d_38,
            l_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_14_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_14_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_38 = (
            l_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_14_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_14_modules_1_modules_bn_parameters_bias_ = None
        input_30 = torch.nn.functional.relu(batch_norm_38, inplace=True)
        batch_norm_38 = None
        conv2d_39 = torch.conv2d(
            input_30,
            l_self_modules_model_modules_14_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_30 = (
            l_self_modules_model_modules_14_modules_2_modules_conv_parameters_weight_
        ) = None
        batch_norm_39 = torch.nn.functional.batch_norm(
            conv2d_39,
            l_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_14_modules_2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_14_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_39 = (
            l_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_14_modules_2_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_14_modules_2_modules_bn_parameters_bias_ = None
        input_31 = torch.nn.functional.relu(batch_norm_39, inplace=True)
        batch_norm_39 = None
        conv2d_40 = torch.conv2d(
            input_31,
            l_self_modules_model_modules_14_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_31 = (
            l_self_modules_model_modules_14_modules_3_modules_conv_parameters_weight_
        ) = None
        batch_norm_40 = torch.nn.functional.batch_norm(
            conv2d_40,
            l_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_14_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_14_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_40 = (
            l_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_14_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_14_modules_3_modules_bn_parameters_bias_ = None
        input_32 = torch.nn.functional.relu(batch_norm_40, inplace=True)
        batch_norm_40 = None
        conv2d_41 = torch.conv2d(
            input_32,
            l_self_modules_model_modules_14_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_32 = (
            l_self_modules_model_modules_14_modules_4_modules_conv_parameters_weight_
        ) = None
        batch_norm_41 = torch.nn.functional.batch_norm(
            conv2d_41,
            l_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_14_modules_4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_14_modules_4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_41 = (
            l_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_14_modules_4_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_14_modules_4_modules_bn_parameters_bias_ = None
        input_33 = torch.nn.functional.relu(batch_norm_41, inplace=True)
        batch_norm_41 = None
        conv2d_42 = torch.conv2d(
            input_33,
            l_self_modules_model_modules_14_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_33 = (
            l_self_modules_model_modules_14_modules_5_modules_conv_parameters_weight_
        ) = None
        batch_norm_42 = torch.nn.functional.batch_norm(
            conv2d_42,
            l_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_14_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_14_modules_5_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_42 = (
            l_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_14_modules_5_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_14_modules_5_modules_bn_parameters_bias_ = None
        input_34 = torch.nn.functional.relu(batch_norm_42, inplace=True)
        batch_norm_42 = None
        conv2d_43 = torch.conv2d(
            input_34,
            l_self_modules_model_modules_15_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_34 = (
            l_self_modules_model_modules_15_modules_conv_parameters_weight_
        ) = None
        batch_norm_43 = torch.nn.functional.batch_norm(
            conv2d_43,
            l_self_modules_model_modules_15_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_43 = (
            l_self_modules_model_modules_15_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_15_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_15_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_15_modules_bn_parameters_bias_ = None
        x_10 = torch.nn.functional.relu(batch_norm_43, inplace=True)
        batch_norm_43 = None
        x_11 = torch.conv_transpose2d(
            x_10,
            l_self_modules_model_modules_16_parameters_weight_,
            l_self_modules_model_modules_16_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        l_self_modules_model_modules_16_parameters_weight_ = (
            l_self_modules_model_modules_16_parameters_bias_
        ) = None
        x_12 = torch.cat([x_11, input_12], 1)
        x_11 = input_12 = None
        conv2d_44 = torch.conv2d(
            x_12,
            l_self_modules_model_modules_18_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_model_modules_18_modules_conv_parameters_weight_ = None
        batch_norm_44 = torch.nn.functional.batch_norm(
            conv2d_44,
            l_self_modules_model_modules_18_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_18_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_18_modules_bn_parameters_weight_,
            l_self_modules_model_modules_18_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_44 = (
            l_self_modules_model_modules_18_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_18_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_18_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_18_modules_bn_parameters_bias_ = None
        x_13 = torch.nn.functional.relu(batch_norm_44, inplace=True)
        batch_norm_44 = None
        conv2d_45 = torch.conv2d(
            x_13,
            l_self_modules_model_modules_19_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_13 = (
            l_self_modules_model_modules_19_modules_0_modules_conv_parameters_weight_
        ) = None
        batch_norm_45 = torch.nn.functional.batch_norm(
            conv2d_45,
            l_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_45 = (
            l_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_19_modules_0_modules_bn_parameters_bias_ = None
        input_35 = torch.nn.functional.relu(batch_norm_45, inplace=True)
        batch_norm_45 = None
        conv2d_46 = torch.conv2d(
            input_35,
            l_self_modules_model_modules_19_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_35 = (
            l_self_modules_model_modules_19_modules_1_modules_conv_parameters_weight_
        ) = None
        batch_norm_46 = torch.nn.functional.batch_norm(
            conv2d_46,
            l_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_46 = (
            l_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_19_modules_1_modules_bn_parameters_bias_ = None
        input_36 = torch.nn.functional.relu(batch_norm_46, inplace=True)
        batch_norm_46 = None
        conv2d_47 = torch.conv2d(
            input_36,
            l_self_modules_model_modules_19_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_36 = (
            l_self_modules_model_modules_19_modules_2_modules_conv_parameters_weight_
        ) = None
        batch_norm_47 = torch.nn.functional.batch_norm(
            conv2d_47,
            l_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_47 = (
            l_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_2_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_19_modules_2_modules_bn_parameters_bias_ = None
        input_37 = torch.nn.functional.relu(batch_norm_47, inplace=True)
        batch_norm_47 = None
        conv2d_48 = torch.conv2d(
            input_37,
            l_self_modules_model_modules_19_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_37 = (
            l_self_modules_model_modules_19_modules_3_modules_conv_parameters_weight_
        ) = None
        batch_norm_48 = torch.nn.functional.batch_norm(
            conv2d_48,
            l_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_48 = (
            l_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_19_modules_3_modules_bn_parameters_bias_ = None
        input_38 = torch.nn.functional.relu(batch_norm_48, inplace=True)
        batch_norm_48 = None
        conv2d_49 = torch.conv2d(
            input_38,
            l_self_modules_model_modules_19_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_38 = (
            l_self_modules_model_modules_19_modules_4_modules_conv_parameters_weight_
        ) = None
        batch_norm_49 = torch.nn.functional.batch_norm(
            conv2d_49,
            l_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_49 = (
            l_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_4_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_19_modules_4_modules_bn_parameters_bias_ = None
        input_39 = torch.nn.functional.relu(batch_norm_49, inplace=True)
        batch_norm_49 = None
        conv2d_50 = torch.conv2d(
            input_39,
            l_self_modules_model_modules_19_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_39 = (
            l_self_modules_model_modules_19_modules_5_modules_conv_parameters_weight_
        ) = None
        batch_norm_50 = torch.nn.functional.batch_norm(
            conv2d_50,
            l_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_5_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_50 = (
            l_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_5_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_19_modules_5_modules_bn_parameters_bias_ = None
        input_40 = torch.nn.functional.relu(batch_norm_50, inplace=True)
        batch_norm_50 = None
        conv2d_51 = torch.conv2d(
            input_40,
            l_self_modules_model_modules_20_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_20_modules_conv_parameters_weight_ = None
        batch_norm_51 = torch.nn.functional.batch_norm(
            conv2d_51,
            l_self_modules_model_modules_20_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_20_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_20_modules_bn_parameters_weight_,
            l_self_modules_model_modules_20_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_51 = (
            l_self_modules_model_modules_20_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_20_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_20_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_20_modules_bn_parameters_bias_ = None
        x_14 = torch.nn.functional.relu(batch_norm_51, inplace=True)
        batch_norm_51 = None
        x_15 = torch.cat([x_14, x_10], 1)
        x_14 = x_10 = None
        conv2d_52 = torch.conv2d(
            x_15,
            l_self_modules_model_modules_22_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_model_modules_22_modules_conv_parameters_weight_ = None
        batch_norm_52 = torch.nn.functional.batch_norm(
            conv2d_52,
            l_self_modules_model_modules_22_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_22_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_22_modules_bn_parameters_weight_,
            l_self_modules_model_modules_22_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_52 = (
            l_self_modules_model_modules_22_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_22_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_22_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_22_modules_bn_parameters_bias_ = None
        x_16 = torch.nn.functional.relu(batch_norm_52, inplace=True)
        batch_norm_52 = None
        conv2d_53 = torch.conv2d(
            x_16,
            l_self_modules_model_modules_23_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_16 = (
            l_self_modules_model_modules_23_modules_0_modules_conv_parameters_weight_
        ) = None
        batch_norm_53 = torch.nn.functional.batch_norm(
            conv2d_53,
            l_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_53 = (
            l_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_23_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_23_modules_0_modules_bn_parameters_bias_ = None
        input_41 = torch.nn.functional.relu(batch_norm_53, inplace=True)
        batch_norm_53 = None
        conv2d_54 = torch.conv2d(
            input_41,
            l_self_modules_model_modules_23_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_41 = (
            l_self_modules_model_modules_23_modules_1_modules_conv_parameters_weight_
        ) = None
        batch_norm_54 = torch.nn.functional.batch_norm(
            conv2d_54,
            l_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_54 = (
            l_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_23_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_23_modules_1_modules_bn_parameters_bias_ = None
        input_42 = torch.nn.functional.relu(batch_norm_54, inplace=True)
        batch_norm_54 = None
        conv2d_55 = torch.conv2d(
            input_42,
            l_self_modules_model_modules_23_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_42 = (
            l_self_modules_model_modules_23_modules_2_modules_conv_parameters_weight_
        ) = None
        batch_norm_55 = torch.nn.functional.batch_norm(
            conv2d_55,
            l_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_55 = (
            l_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_23_modules_2_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_23_modules_2_modules_bn_parameters_bias_ = None
        input_43 = torch.nn.functional.relu(batch_norm_55, inplace=True)
        batch_norm_55 = None
        conv2d_56 = torch.conv2d(
            input_43,
            l_self_modules_model_modules_23_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_43 = (
            l_self_modules_model_modules_23_modules_3_modules_conv_parameters_weight_
        ) = None
        batch_norm_56 = torch.nn.functional.batch_norm(
            conv2d_56,
            l_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_56 = (
            l_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_23_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_23_modules_3_modules_bn_parameters_bias_ = None
        input_44 = torch.nn.functional.relu(batch_norm_56, inplace=True)
        batch_norm_56 = None
        conv2d_57 = torch.conv2d(
            input_44,
            l_self_modules_model_modules_23_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_44 = (
            l_self_modules_model_modules_23_modules_4_modules_conv_parameters_weight_
        ) = None
        batch_norm_57 = torch.nn.functional.batch_norm(
            conv2d_57,
            l_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_57 = (
            l_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_23_modules_4_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_23_modules_4_modules_bn_parameters_bias_ = None
        input_45 = torch.nn.functional.relu(batch_norm_57, inplace=True)
        batch_norm_57 = None
        conv2d_58 = torch.conv2d(
            input_45,
            l_self_modules_model_modules_23_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_45 = (
            l_self_modules_model_modules_23_modules_5_modules_conv_parameters_weight_
        ) = None
        batch_norm_58 = torch.nn.functional.batch_norm(
            conv2d_58,
            l_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_23_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_23_modules_5_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_58 = (
            l_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_23_modules_5_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_23_modules_5_modules_bn_parameters_bias_ = None
        input_46 = torch.nn.functional.relu(batch_norm_58, inplace=True)
        batch_norm_58 = None
        conv2d_59 = torch.conv2d(
            input_46,
            l_self_modules_model_modules_24_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_24_modules_conv_parameters_weight_ = None
        batch_norm_59 = torch.nn.functional.batch_norm(
            conv2d_59,
            l_self_modules_model_modules_24_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_24_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_24_modules_bn_parameters_weight_,
            l_self_modules_model_modules_24_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_59 = (
            l_self_modules_model_modules_24_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_24_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_24_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_24_modules_bn_parameters_bias_ = None
        x_17 = torch.nn.functional.relu(batch_norm_59, inplace=True)
        batch_norm_59 = None
        x_18 = torch.cat([x_17, x_6], 1)
        x_17 = x_6 = None
        conv2d_60 = torch.conv2d(
            x_18,
            l_self_modules_model_modules_26_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_model_modules_26_modules_conv_parameters_weight_ = None
        batch_norm_60 = torch.nn.functional.batch_norm(
            conv2d_60,
            l_self_modules_model_modules_26_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_26_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_26_modules_bn_parameters_weight_,
            l_self_modules_model_modules_26_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_60 = (
            l_self_modules_model_modules_26_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_26_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_26_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_26_modules_bn_parameters_bias_ = None
        x_19 = torch.nn.functional.relu(batch_norm_60, inplace=True)
        batch_norm_60 = None
        conv2d_61 = torch.conv2d(
            x_19,
            l_self_modules_model_modules_27_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_19 = (
            l_self_modules_model_modules_27_modules_0_modules_conv_parameters_weight_
        ) = None
        batch_norm_61 = torch.nn.functional.batch_norm(
            conv2d_61,
            l_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_27_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_27_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_61 = (
            l_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_27_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_27_modules_0_modules_bn_parameters_bias_ = None
        input_47 = torch.nn.functional.relu(batch_norm_61, inplace=True)
        batch_norm_61 = None
        conv2d_62 = torch.conv2d(
            input_47,
            l_self_modules_model_modules_27_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_47 = (
            l_self_modules_model_modules_27_modules_1_modules_conv_parameters_weight_
        ) = None
        batch_norm_62 = torch.nn.functional.batch_norm(
            conv2d_62,
            l_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_27_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_27_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_62 = (
            l_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_27_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_27_modules_1_modules_bn_parameters_bias_ = None
        input_48 = torch.nn.functional.relu(batch_norm_62, inplace=True)
        batch_norm_62 = None
        conv2d_63 = torch.conv2d(
            input_48,
            l_self_modules_model_modules_27_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_48 = (
            l_self_modules_model_modules_27_modules_2_modules_conv_parameters_weight_
        ) = None
        batch_norm_63 = torch.nn.functional.batch_norm(
            conv2d_63,
            l_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_27_modules_2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_27_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_63 = (
            l_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_27_modules_2_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_27_modules_2_modules_bn_parameters_bias_ = None
        input_49 = torch.nn.functional.relu(batch_norm_63, inplace=True)
        batch_norm_63 = None
        conv2d_64 = torch.conv2d(
            input_49,
            l_self_modules_model_modules_27_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_49 = (
            l_self_modules_model_modules_27_modules_3_modules_conv_parameters_weight_
        ) = None
        batch_norm_64 = torch.nn.functional.batch_norm(
            conv2d_64,
            l_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_27_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_27_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_64 = (
            l_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_27_modules_3_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_27_modules_3_modules_bn_parameters_bias_ = None
        input_50 = torch.nn.functional.relu(batch_norm_64, inplace=True)
        batch_norm_64 = None
        conv2d_65 = torch.conv2d(
            input_50,
            l_self_modules_model_modules_27_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_50 = (
            l_self_modules_model_modules_27_modules_4_modules_conv_parameters_weight_
        ) = None
        batch_norm_65 = torch.nn.functional.batch_norm(
            conv2d_65,
            l_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_27_modules_4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_27_modules_4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_65 = (
            l_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_27_modules_4_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_27_modules_4_modules_bn_parameters_bias_ = None
        input_51 = torch.nn.functional.relu(batch_norm_65, inplace=True)
        batch_norm_65 = None
        conv2d_66 = torch.conv2d(
            input_51,
            l_self_modules_model_modules_27_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_51 = (
            l_self_modules_model_modules_27_modules_5_modules_conv_parameters_weight_
        ) = None
        batch_norm_66 = torch.nn.functional.batch_norm(
            conv2d_66,
            l_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_27_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_27_modules_5_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_66 = (
            l_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_27_modules_5_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_27_modules_5_modules_bn_parameters_bias_ = None
        input_52 = torch.nn.functional.relu(batch_norm_66, inplace=True)
        batch_norm_66 = None
        conv2d_67 = torch.conv2d(
            input_40,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_67 = torch.nn.functional.batch_norm(
            conv2d_67,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_67 = l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_53 = torch.nn.functional.relu(batch_norm_67, inplace=True)
        batch_norm_67 = None
        conv2d_68 = torch.conv2d(
            input_53,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_53 = l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_68 = torch.nn.functional.batch_norm(
            conv2d_68,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_68 = l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_54 = torch.nn.functional.relu(batch_norm_68, inplace=True)
        batch_norm_68 = None
        input_55 = torch.conv2d(
            input_54,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_54 = l_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_weight_ = l_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_bias_ = (None)
        conv2d_70 = torch.conv2d(
            input_40,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_40 = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_69 = torch.nn.functional.batch_norm(
            conv2d_70,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_70 = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_56 = torch.nn.functional.relu(batch_norm_69, inplace=True)
        batch_norm_69 = None
        conv2d_71 = torch.conv2d(
            input_56,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_56 = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_70 = torch.nn.functional.batch_norm(
            conv2d_71,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_71 = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_57 = torch.nn.functional.relu(batch_norm_70, inplace=True)
        batch_norm_70 = None
        input_58 = torch.conv2d(
            input_57,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_57 = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_weight_ = l_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_bias_ = (None)
        xi = torch.cat((input_55, input_58), 1)
        input_55 = input_58 = None
        conv2d_73 = torch.conv2d(
            input_46,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_71 = torch.nn.functional.batch_norm(
            conv2d_73,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_73 = l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_59 = torch.nn.functional.relu(batch_norm_71, inplace=True)
        batch_norm_71 = None
        conv2d_74 = torch.conv2d(
            input_59,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_59 = l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_72 = torch.nn.functional.batch_norm(
            conv2d_74,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_74 = l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_60 = torch.nn.functional.relu(batch_norm_72, inplace=True)
        batch_norm_72 = None
        input_61 = torch.conv2d(
            input_60,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_60 = l_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_weight_ = l_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_bias_ = (None)
        conv2d_76 = torch.conv2d(
            input_46,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_46 = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_73 = torch.nn.functional.batch_norm(
            conv2d_76,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_76 = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_62 = torch.nn.functional.relu(batch_norm_73, inplace=True)
        batch_norm_73 = None
        conv2d_77 = torch.conv2d(
            input_62,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_62 = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_74 = torch.nn.functional.batch_norm(
            conv2d_77,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_77 = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_63 = torch.nn.functional.relu(batch_norm_74, inplace=True)
        batch_norm_74 = None
        input_64 = torch.conv2d(
            input_63,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_63 = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_weight_ = l_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_bias_ = (None)
        xi_1 = torch.cat((input_61, input_64), 1)
        input_61 = input_64 = None
        conv2d_79 = torch.conv2d(
            input_52,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_75 = torch.nn.functional.batch_norm(
            conv2d_79,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_79 = l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        input_65 = torch.nn.functional.relu(batch_norm_75, inplace=True)
        batch_norm_75 = None
        conv2d_80 = torch.conv2d(
            input_65,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_65 = l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_76 = torch.nn.functional.batch_norm(
            conv2d_80,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_80 = l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        input_66 = torch.nn.functional.relu(batch_norm_76, inplace=True)
        batch_norm_76 = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_66 = l_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_weight_ = l_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_bias_ = (None)
        conv2d_82 = torch.conv2d(
            input_52,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_52 = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_77 = torch.nn.functional.batch_norm(
            conv2d_82,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_82 = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        input_68 = torch.nn.functional.relu(batch_norm_77, inplace=True)
        batch_norm_77 = None
        conv2d_83 = torch.conv2d(
            input_68,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_68 = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_78 = torch.nn.functional.batch_norm(
            conv2d_83,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_83 = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        input_69 = torch.nn.functional.relu(batch_norm_78, inplace=True)
        batch_norm_78 = None
        input_70 = torch.conv2d(
            input_69,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_weight_,
            l_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_69 = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_weight_ = l_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_bias_ = (None)
        xi_2 = torch.cat((input_67, input_70), 1)
        input_67 = input_70 = None
        view = xi.view(1, 144, -1)
        view_1 = xi_1.view(1, 144, -1)
        view_2 = xi_2.view(1, 144, -1)
        x_cat = torch.cat([view, view_1, view_2], 2)
        view = view_1 = view_2 = None
        x_20 = l_self_modules_model_modules_28_stride[0]
        x_21 = l_self_modules_model_modules_28_stride[1]
        x_22 = l_self_modules_model_modules_28_stride[2]
        l_self_modules_model_modules_28_stride = None
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
        _local_scalar_dense = torch.ops.aten._local_scalar_dense(x_20)
        x_20 = None
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
        _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense(x_21)
        x_21 = None
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
        _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense(x_22)
        x_22 = None
        full_2 = torch.full(
            (400, 1),
            _local_scalar_dense_2,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        _local_scalar_dense_2 = None
        x_23 = torch.cat([view_3, view_4, view_5])
        view_3 = view_4 = view_5 = None
        x_24 = torch.cat([full, full_1, full_2])
        full = full_1 = full_2 = None
        transpose = x_23.transpose(0, 1)
        x_23 = None
        transpose_1 = x_24.transpose(0, 1)
        x_24 = None
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
        conv2d_85 = torch.conv2d(
            softmax,
            l_self_modules_model_modules_28_modules_dfl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        softmax = (
            l_self_modules_model_modules_28_modules_dfl_modules_conv_parameters_weight_
        ) = None
        view_7 = conv2d_85.view(1, 4, 8400)
        conv2d_85 = None
        unsqueeze = transpose.unsqueeze(0)
        chunk = view_7.chunk(2, 1)
        view_7 = None
        lt = chunk[0]
        rb = chunk[1]
        chunk = None
        x1y1 = unsqueeze - lt
        lt = None
        x2y2 = unsqueeze + rb
        unsqueeze = rb = None
        add_7 = x1y1 + x2y2
        c_xy = add_7 / 2
        add_7 = None
        wh = x2y2 - x1y1
        x2y2 = x1y1 = None
        cat_11 = torch.cat((c_xy, wh), 1)
        c_xy = wh = None
        dbox = cat_11 * transpose_1
        cat_11 = None
        sigmoid = cls.sigmoid()
        cls = None
        y = torch.cat((dbox, sigmoid), 1)
        dbox = sigmoid = None
        return (y, xi, xi_1, xi_2, transpose_1, transpose)
