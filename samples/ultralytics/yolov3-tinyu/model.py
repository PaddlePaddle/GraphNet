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
        L_self_modules_model_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_10_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_10_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_10_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_13_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_13_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_13_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_14_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_14_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_14_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_15_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_15_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_15_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_16_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_16_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_16_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_19_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_19_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_19_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_0_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv2_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_modules_cv3_modules_1_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_20_stride: torch.Tensor,
        L_self_modules_model_modules_20_modules_dfl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_model_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_8_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_8_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_8_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_8_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_8_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_8_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_14_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_14_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_14_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_14_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_14_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_14_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_14_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_14_modules_bn_parameters_bias_
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
        l_self_modules_model_modules_16_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_16_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_16_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_16_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_16_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_16_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_16_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_16_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_16_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_16_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_19_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_19_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_19_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_19_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_19_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_19_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_19_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_19_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_2_parameters_weight_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_2_parameters_weight_
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_2_parameters_bias_ = L_self_modules_model_modules_20_modules_cv2_modules_0_modules_2_parameters_bias_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_2_parameters_weight_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_2_parameters_weight_
        l_self_modules_model_modules_20_modules_cv3_modules_0_modules_2_parameters_bias_ = L_self_modules_model_modules_20_modules_cv3_modules_0_modules_2_parameters_bias_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_2_parameters_weight_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_2_parameters_weight_
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_2_parameters_bias_ = L_self_modules_model_modules_20_modules_cv2_modules_1_modules_2_parameters_bias_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_2_parameters_weight_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_2_parameters_weight_
        l_self_modules_model_modules_20_modules_cv3_modules_1_modules_2_parameters_bias_ = L_self_modules_model_modules_20_modules_cv3_modules_1_modules_2_parameters_bias_
        l_self_modules_model_modules_20_stride = L_self_modules_model_modules_20_stride
        l_self_modules_model_modules_20_modules_dfl_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_20_modules_dfl_modules_conv_parameters_weight_
        )
        conv2d = torch.conv2d(
            l_x_,
            l_self_modules_model_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
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
        x_1 = torch.nn.functional.max_pool2d(
            x, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x = None
        conv2d_1 = torch.conv2d(
            x_1,
            l_self_modules_model_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_1 = l_self_modules_model_modules_2_modules_conv_parameters_weight_ = None
        batch_norm_1 = torch.nn.functional.batch_norm(
            conv2d_1,
            l_self_modules_model_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_1 = (
            l_self_modules_model_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_2_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.silu(batch_norm_1, inplace=True)
        batch_norm_1 = None
        x_3 = torch.nn.functional.max_pool2d(
            x_2, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_2 = None
        conv2d_2 = torch.conv2d(
            x_3,
            l_self_modules_model_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_model_modules_4_modules_conv_parameters_weight_ = None
        batch_norm_2 = torch.nn.functional.batch_norm(
            conv2d_2,
            l_self_modules_model_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_2 = (
            l_self_modules_model_modules_4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_4_modules_bn_parameters_bias_ = None
        x_4 = torch.nn.functional.silu(batch_norm_2, inplace=True)
        batch_norm_2 = None
        x_5 = torch.nn.functional.max_pool2d(
            x_4, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_4 = None
        conv2d_3 = torch.conv2d(
            x_5,
            l_self_modules_model_modules_6_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_model_modules_6_modules_conv_parameters_weight_ = None
        batch_norm_3 = torch.nn.functional.batch_norm(
            conv2d_3,
            l_self_modules_model_modules_6_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_3 = (
            l_self_modules_model_modules_6_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_6_modules_bn_parameters_bias_ = None
        x_6 = torch.nn.functional.silu(batch_norm_3, inplace=True)
        batch_norm_3 = None
        x_7 = torch.nn.functional.max_pool2d(
            x_6, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_6 = None
        conv2d_4 = torch.conv2d(
            x_7,
            l_self_modules_model_modules_8_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_model_modules_8_modules_conv_parameters_weight_ = None
        batch_norm_4 = torch.nn.functional.batch_norm(
            conv2d_4,
            l_self_modules_model_modules_8_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_4 = (
            l_self_modules_model_modules_8_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_8_modules_bn_parameters_bias_ = None
        x_8 = torch.nn.functional.silu(batch_norm_4, inplace=True)
        batch_norm_4 = None
        x_9 = torch.nn.functional.max_pool2d(
            x_8, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        conv2d_5 = torch.conv2d(
            x_9,
            l_self_modules_model_modules_10_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_model_modules_10_modules_conv_parameters_weight_ = None
        batch_norm_5 = torch.nn.functional.batch_norm(
            conv2d_5,
            l_self_modules_model_modules_10_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_10_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_10_modules_bn_parameters_weight_,
            l_self_modules_model_modules_10_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_5 = (
            l_self_modules_model_modules_10_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_10_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_10_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_10_modules_bn_parameters_bias_ = None
        x_10 = torch.nn.functional.silu(batch_norm_5, inplace=True)
        batch_norm_5 = None
        x_11 = torch._C._nn.pad(x_10, (0, 1, 0, 1), "constant", 0.0)
        x_10 = None
        x_12 = torch.nn.functional.max_pool2d(
            x_11, 2, 1, 0, 1, ceil_mode=False, return_indices=False
        )
        x_11 = None
        conv2d_6 = torch.conv2d(
            x_12,
            l_self_modules_model_modules_13_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_model_modules_13_modules_conv_parameters_weight_ = None
        batch_norm_6 = torch.nn.functional.batch_norm(
            conv2d_6,
            l_self_modules_model_modules_13_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_13_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_13_modules_bn_parameters_weight_,
            l_self_modules_model_modules_13_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_6 = (
            l_self_modules_model_modules_13_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_13_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_13_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_13_modules_bn_parameters_bias_ = None
        x_13 = torch.nn.functional.silu(batch_norm_6, inplace=True)
        batch_norm_6 = None
        conv2d_7 = torch.conv2d(
            x_13,
            l_self_modules_model_modules_14_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_model_modules_14_modules_conv_parameters_weight_ = None
        batch_norm_7 = torch.nn.functional.batch_norm(
            conv2d_7,
            l_self_modules_model_modules_14_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_14_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_14_modules_bn_parameters_weight_,
            l_self_modules_model_modules_14_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_7 = (
            l_self_modules_model_modules_14_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_14_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_14_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_14_modules_bn_parameters_bias_ = None
        x_14 = torch.nn.functional.silu(batch_norm_7, inplace=True)
        batch_norm_7 = None
        conv2d_8 = torch.conv2d(
            x_14,
            l_self_modules_model_modules_15_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_15_modules_conv_parameters_weight_ = None
        batch_norm_8 = torch.nn.functional.batch_norm(
            conv2d_8,
            l_self_modules_model_modules_15_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_15_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_15_modules_bn_parameters_weight_,
            l_self_modules_model_modules_15_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_8 = (
            l_self_modules_model_modules_15_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_15_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_15_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_15_modules_bn_parameters_bias_ = None
        x_15 = torch.nn.functional.silu(batch_norm_8, inplace=True)
        batch_norm_8 = None
        conv2d_9 = torch.conv2d(
            x_14,
            l_self_modules_model_modules_16_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_model_modules_16_modules_conv_parameters_weight_ = None
        batch_norm_9 = torch.nn.functional.batch_norm(
            conv2d_9,
            l_self_modules_model_modules_16_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_16_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_16_modules_bn_parameters_weight_,
            l_self_modules_model_modules_16_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_9 = (
            l_self_modules_model_modules_16_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_16_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_16_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_16_modules_bn_parameters_bias_ = None
        x_16 = torch.nn.functional.silu(batch_norm_9, inplace=True)
        batch_norm_9 = None
        x_17 = torch.nn.functional.interpolate(
            x_16, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        x_16 = None
        x_18 = torch.cat([x_17, x_8], 1)
        x_17 = x_8 = None
        conv2d_10 = torch.conv2d(
            x_18,
            l_self_modules_model_modules_19_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_model_modules_19_modules_conv_parameters_weight_ = None
        batch_norm_10 = torch.nn.functional.batch_norm(
            conv2d_10,
            l_self_modules_model_modules_19_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_19_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_19_modules_bn_parameters_weight_,
            l_self_modules_model_modules_19_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_10 = (
            l_self_modules_model_modules_19_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_19_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_19_modules_bn_parameters_weight_
        ) = l_self_modules_model_modules_19_modules_bn_parameters_bias_ = None
        x_19 = torch.nn.functional.silu(batch_norm_10, inplace=True)
        batch_norm_10 = None
        conv2d_11 = torch.conv2d(
            x_19,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_11 = torch.nn.functional.batch_norm(
            conv2d_11,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_11 = l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_20_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_1 = torch.nn.functional.silu(batch_norm_11, inplace=True)
        batch_norm_11 = None
        conv2d_12 = torch.conv2d(
            input_1,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_12 = torch.nn.functional.batch_norm(
            conv2d_12,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_12 = l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_20_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_2 = torch.nn.functional.silu(batch_norm_12, inplace=True)
        batch_norm_12 = None
        input_3 = torch.conv2d(
            input_2,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_2_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv2_modules_0_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_model_modules_20_modules_cv2_modules_0_modules_2_parameters_weight_ = l_self_modules_model_modules_20_modules_cv2_modules_0_modules_2_parameters_bias_ = (None)
        conv2d_14 = torch.conv2d(
            x_19,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_13 = torch.nn.functional.batch_norm(
            conv2d_14,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_14 = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        input_4 = torch.nn.functional.silu(batch_norm_13, inplace=True)
        batch_norm_13 = None
        conv2d_15 = torch.conv2d(
            input_4,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_14 = torch.nn.functional.batch_norm(
            conv2d_15,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_15 = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        input_5 = torch.nn.functional.silu(batch_norm_14, inplace=True)
        batch_norm_14 = None
        input_6 = torch.conv2d(
            input_5,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_2_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv3_modules_0_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_2_parameters_weight_ = l_self_modules_model_modules_20_modules_cv3_modules_0_modules_2_parameters_bias_ = (None)
        xi = torch.cat((input_3, input_6), 1)
        input_3 = input_6 = None
        conv2d_17 = torch.conv2d(
            x_15,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        batch_norm_15 = torch.nn.functional.batch_norm(
            conv2d_17,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_17 = l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_20_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_7 = torch.nn.functional.silu(batch_norm_15, inplace=True)
        batch_norm_15 = None
        conv2d_18 = torch.conv2d(
            input_7,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_16 = torch.nn.functional.batch_norm(
            conv2d_18,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_18 = l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_20_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_8 = torch.nn.functional.silu(batch_norm_16, inplace=True)
        batch_norm_16 = None
        input_9 = torch.conv2d(
            input_8,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_2_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv2_modules_1_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_8 = l_self_modules_model_modules_20_modules_cv2_modules_1_modules_2_parameters_weight_ = l_self_modules_model_modules_20_modules_cv2_modules_1_modules_2_parameters_bias_ = (None)
        conv2d_20 = torch.conv2d(
            x_15,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        batch_norm_17 = torch.nn.functional.batch_norm(
            conv2d_20,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_20 = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        input_10 = torch.nn.functional.silu(batch_norm_17, inplace=True)
        batch_norm_17 = None
        conv2d_21 = torch.conv2d(
            input_10,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_10 = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        batch_norm_18 = torch.nn.functional.batch_norm(
            conv2d_21,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        conv2d_21 = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        input_11 = torch.nn.functional.silu(batch_norm_18, inplace=True)
        batch_norm_18 = None
        input_12 = torch.conv2d(
            input_11,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_2_parameters_weight_,
            l_self_modules_model_modules_20_modules_cv3_modules_1_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_11 = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_2_parameters_weight_ = l_self_modules_model_modules_20_modules_cv3_modules_1_modules_2_parameters_bias_ = (None)
        xi_1 = torch.cat((input_9, input_12), 1)
        input_9 = input_12 = None
        view = xi.view(1, 144, -1)
        view_1 = xi_1.view(1, 144, -1)
        x_cat = torch.cat([view, view_1], 2)
        view = view_1 = None
        x_20 = l_self_modules_model_modules_20_stride[0]
        x_21 = l_self_modules_model_modules_20_stride[1]
        l_self_modules_model_modules_20_stride = None
        arange = torch.arange(
            end=40, device=device(type="cuda", index=0), dtype=torch.float32
        )
        sx = arange + 0.5
        arange = None
        arange_1 = torch.arange(
            end=40, device=device(type="cuda", index=0), dtype=torch.float32
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
        view_2 = stack.view(-1, 2)
        stack = None
        _local_scalar_dense = torch.ops.aten._local_scalar_dense(x_20)
        x_20 = None
        full = torch.full(
            (1600, 1),
            _local_scalar_dense,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        _local_scalar_dense = None
        arange_2 = torch.arange(
            end=20, device=device(type="cuda", index=0), dtype=torch.float32
        )
        sx_2 = arange_2 + 0.5
        arange_2 = None
        arange_3 = torch.arange(
            end=20, device=device(type="cuda", index=0), dtype=torch.float32
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
        view_3 = stack_1.view(-1, 2)
        stack_1 = None
        _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense(x_21)
        x_21 = None
        full_1 = torch.full(
            (400, 1),
            _local_scalar_dense_1,
            dtype=torch.float32,
            device=device(type="cuda", index=0),
        )
        _local_scalar_dense_1 = None
        x_22 = torch.cat([view_2, view_3])
        view_2 = view_3 = None
        x_23 = torch.cat([full, full_1])
        full = full_1 = None
        transpose = x_22.transpose(0, 1)
        x_22 = None
        transpose_1 = x_23.transpose(0, 1)
        x_23 = None
        split = x_cat.split((64, 80), 1)
        x_cat = None
        box = split[0]
        cls = split[1]
        split = None
        view_4 = box.view(1, 4, 16, 2000)
        box = None
        transpose_2 = view_4.transpose(2, 1)
        view_4 = None
        softmax = transpose_2.softmax(1)
        transpose_2 = None
        conv2d_23 = torch.conv2d(
            softmax,
            l_self_modules_model_modules_20_modules_dfl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        softmax = (
            l_self_modules_model_modules_20_modules_dfl_modules_conv_parameters_weight_
        ) = None
        view_5 = conv2d_23.view(1, 4, 2000)
        conv2d_23 = None
        unsqueeze = transpose.unsqueeze(0)
        chunk = view_5.chunk(2, 1)
        view_5 = None
        lt = chunk[0]
        rb = chunk[1]
        chunk = None
        x1y1 = unsqueeze - lt
        lt = None
        x2y2 = unsqueeze + rb
        unsqueeze = rb = None
        add_5 = x1y1 + x2y2
        c_xy = add_5 / 2
        add_5 = None
        wh = x2y2 - x1y1
        x2y2 = x1y1 = None
        cat_6 = torch.cat((c_xy, wh), 1)
        c_xy = wh = None
        dbox = cat_6 * transpose_1
        cat_6 = None
        sigmoid = cls.sigmoid()
        cls = None
        y = torch.cat((dbox, sigmoid), 1)
        dbox = sigmoid = None
        return (y, xi, xi_1, transpose_1, transpose)
