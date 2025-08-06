import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch1_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch1_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch1_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch1_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch1_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch1_1a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch1_1a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch1_1a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch1_1a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch1_1a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch1_1b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch1_1b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch1_1b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch1_1b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch1_1b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch2_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch2_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch2_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch2_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch2_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch2_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_3a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_3a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch2_3a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch2_3a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_3a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_3b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_3b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch2_3b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch2_3b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch2_3b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch1_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch1_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch1_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch1_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch1_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch1_1a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch1_1a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch1_1a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch1_1a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch1_1a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch1_1b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch1_1b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch1_1b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch1_1b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch1_1b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch2_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch2_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch2_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch2_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch2_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch2_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_3a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_3a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch2_3a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch2_3a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_3a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_3b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_3b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch2_3b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch2_3b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch2_3b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch1_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch1_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch1_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch1_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch1_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch1_1a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch1_1a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch1_1a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch1_1a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch1_1a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch1_1b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch1_1b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch1_1b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch1_1b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch1_1b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch2_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch2_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch2_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch2_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch2_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch2_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_3a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_3a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch2_3a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch2_3a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_3a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_3b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_3b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch2_3b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch2_3b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch2_3b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_last_linear_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_last_linear_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_features_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_features_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_features_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_features_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_features_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_features_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_features_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_features_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_conv_modules_conv_parameters_weight_ = L_self_modules_features_modules_3_modules_conv_modules_conv_parameters_weight_
        l_self_modules_features_modules_3_modules_conv_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv_modules_bn_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv_modules_bn_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_conv_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_branch0_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_4_modules_branch0_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_4_modules_branch0_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_4_modules_branch0_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_4_modules_branch1_modules_3_modules_conv_parameters_weight_ = L_self_modules_features_modules_4_modules_branch1_modules_3_modules_conv_parameters_weight_
        l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_buffers_running_var_ = L_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_buffers_running_var_
        l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_parameters_weight_ = L_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_parameters_weight_
        l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_parameters_bias_ = L_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_parameters_bias_
        l_self_modules_features_modules_5_modules_conv_modules_conv_parameters_weight_ = L_self_modules_features_modules_5_modules_conv_modules_conv_parameters_weight_
        l_self_modules_features_modules_5_modules_conv_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv_modules_bn_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv_modules_bn_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_conv_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_6_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_6_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_6_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_6_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_6_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_6_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_6_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_6_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_6_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_6_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_6_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_6_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_6_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_6_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_6_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_6_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_6_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_6_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_7_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_7_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_7_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_7_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_7_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_7_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_7_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_7_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_7_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_7_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_7_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_7_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_7_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_7_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_7_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_7_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_7_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_7_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_8_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_8_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_8_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_8_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_8_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_8_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_8_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_8_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_8_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_8_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_8_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_8_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_8_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_8_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_8_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_8_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_8_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_8_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_9_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_9_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_9_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_9_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_9_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_9_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_9_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_9_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_9_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_9_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_9_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_9_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_9_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_9_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_9_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_9_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_9_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_9_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_10_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_10_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_10_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_10_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_10_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_10_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_10_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_10_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_10_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_10_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_10_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_10_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_10_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_10_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_10_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_10_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_branch2_modules_3_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_branch2_modules_3_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_branch2_modules_4_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_branch2_modules_4_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_branch2_modules_3_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_branch2_modules_3_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_branch2_modules_4_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_branch2_modules_4_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_branch2_modules_3_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_branch2_modules_3_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_branch2_modules_4_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_branch2_modules_4_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_branch2_modules_3_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_branch2_modules_3_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_branch2_modules_4_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_branch2_modules_4_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_branch2_modules_3_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_branch2_modules_3_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_branch2_modules_4_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_branch2_modules_4_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_16_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_16_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_16_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_16_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_16_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_16_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_16_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_16_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_16_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_16_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_16_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_16_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_16_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_16_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_16_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_16_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_16_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_16_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_16_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_16_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_16_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_16_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_16_modules_branch2_modules_3_modules_conv_parameters_weight_ = L_self_modules_features_modules_16_modules_branch2_modules_3_modules_conv_parameters_weight_
        l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_buffers_running_var_ = L_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_buffers_running_var_
        l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_parameters_weight_ = L_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_parameters_weight_
        l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_parameters_bias_ = L_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_parameters_bias_
        l_self_modules_features_modules_16_modules_branch2_modules_4_modules_conv_parameters_weight_ = L_self_modules_features_modules_16_modules_branch2_modules_4_modules_conv_parameters_weight_
        l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_buffers_running_var_ = L_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_buffers_running_var_
        l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_parameters_weight_ = L_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_parameters_weight_
        l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_parameters_bias_ = L_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_parameters_bias_
        l_self_modules_features_modules_16_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_16_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_17_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_17_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_17_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_17_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_17_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_17_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_17_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_17_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_17_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_17_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_17_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_17_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_17_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_17_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_17_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_17_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_17_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_17_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_17_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_17_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_17_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_17_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_17_modules_branch2_modules_3_modules_conv_parameters_weight_ = L_self_modules_features_modules_17_modules_branch2_modules_3_modules_conv_parameters_weight_
        l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_buffers_running_var_ = L_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_buffers_running_var_
        l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_parameters_weight_ = L_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_parameters_weight_
        l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_parameters_bias_ = L_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_parameters_bias_
        l_self_modules_features_modules_17_modules_branch2_modules_4_modules_conv_parameters_weight_ = L_self_modules_features_modules_17_modules_branch2_modules_4_modules_conv_parameters_weight_
        l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_buffers_running_var_ = L_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_buffers_running_var_
        l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_parameters_weight_ = L_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_parameters_weight_
        l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_parameters_bias_ = L_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_parameters_bias_
        l_self_modules_features_modules_17_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_17_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_18_modules_branch0_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_18_modules_branch0_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_18_modules_branch0_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_18_modules_branch0_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_18_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_18_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_18_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_18_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_18_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_18_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_18_modules_branch1_modules_3_modules_conv_parameters_weight_ = L_self_modules_features_modules_18_modules_branch1_modules_3_modules_conv_parameters_weight_
        l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_buffers_running_var_ = L_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_buffers_running_var_
        l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_parameters_weight_ = L_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_parameters_weight_
        l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_parameters_bias_ = L_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_parameters_bias_
        l_self_modules_features_modules_19_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_19_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_19_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_19_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_19_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_19_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_19_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_19_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_19_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_19_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_19_modules_branch1_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_19_modules_branch1_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_19_modules_branch1_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_19_modules_branch1_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_19_modules_branch1_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_19_modules_branch1_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_19_modules_branch1_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_19_modules_branch1_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_19_modules_branch1_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_19_modules_branch1_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_19_modules_branch1_1a_modules_conv_parameters_weight_ = L_self_modules_features_modules_19_modules_branch1_1a_modules_conv_parameters_weight_
        l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_19_modules_branch1_1a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_19_modules_branch1_1a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_parameters_weight_ = L_self_modules_features_modules_19_modules_branch1_1a_modules_bn_parameters_weight_
        l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_parameters_bias_ = L_self_modules_features_modules_19_modules_branch1_1a_modules_bn_parameters_bias_
        l_self_modules_features_modules_19_modules_branch1_1b_modules_conv_parameters_weight_ = L_self_modules_features_modules_19_modules_branch1_1b_modules_conv_parameters_weight_
        l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_19_modules_branch1_1b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_19_modules_branch1_1b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_parameters_weight_ = L_self_modules_features_modules_19_modules_branch1_1b_modules_bn_parameters_weight_
        l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_parameters_bias_ = L_self_modules_features_modules_19_modules_branch1_1b_modules_bn_parameters_bias_
        l_self_modules_features_modules_19_modules_branch2_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_19_modules_branch2_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_19_modules_branch2_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_19_modules_branch2_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_19_modules_branch2_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_19_modules_branch2_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_19_modules_branch2_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_19_modules_branch2_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_19_modules_branch2_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_19_modules_branch2_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_19_modules_branch2_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_19_modules_branch2_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_19_modules_branch2_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_19_modules_branch2_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_19_modules_branch2_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_19_modules_branch2_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_19_modules_branch2_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_19_modules_branch2_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_19_modules_branch2_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_19_modules_branch2_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_19_modules_branch2_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_19_modules_branch2_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_19_modules_branch2_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_19_modules_branch2_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_19_modules_branch2_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_19_modules_branch2_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_19_modules_branch2_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_19_modules_branch2_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_19_modules_branch2_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_19_modules_branch2_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_19_modules_branch2_3a_modules_conv_parameters_weight_ = L_self_modules_features_modules_19_modules_branch2_3a_modules_conv_parameters_weight_
        l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_19_modules_branch2_3a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_19_modules_branch2_3a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_parameters_weight_ = L_self_modules_features_modules_19_modules_branch2_3a_modules_bn_parameters_weight_
        l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_parameters_bias_ = L_self_modules_features_modules_19_modules_branch2_3a_modules_bn_parameters_bias_
        l_self_modules_features_modules_19_modules_branch2_3b_modules_conv_parameters_weight_ = L_self_modules_features_modules_19_modules_branch2_3b_modules_conv_parameters_weight_
        l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_19_modules_branch2_3b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_19_modules_branch2_3b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_parameters_weight_ = L_self_modules_features_modules_19_modules_branch2_3b_modules_bn_parameters_weight_
        l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_parameters_bias_ = L_self_modules_features_modules_19_modules_branch2_3b_modules_bn_parameters_bias_
        l_self_modules_features_modules_19_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_19_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_20_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_20_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_20_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_20_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_20_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_20_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_20_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_20_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_20_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_20_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_20_modules_branch1_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_20_modules_branch1_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_20_modules_branch1_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_20_modules_branch1_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_20_modules_branch1_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_20_modules_branch1_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_20_modules_branch1_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_20_modules_branch1_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_20_modules_branch1_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_20_modules_branch1_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_20_modules_branch1_1a_modules_conv_parameters_weight_ = L_self_modules_features_modules_20_modules_branch1_1a_modules_conv_parameters_weight_
        l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_20_modules_branch1_1a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_20_modules_branch1_1a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_parameters_weight_ = L_self_modules_features_modules_20_modules_branch1_1a_modules_bn_parameters_weight_
        l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_parameters_bias_ = L_self_modules_features_modules_20_modules_branch1_1a_modules_bn_parameters_bias_
        l_self_modules_features_modules_20_modules_branch1_1b_modules_conv_parameters_weight_ = L_self_modules_features_modules_20_modules_branch1_1b_modules_conv_parameters_weight_
        l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_20_modules_branch1_1b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_20_modules_branch1_1b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_parameters_weight_ = L_self_modules_features_modules_20_modules_branch1_1b_modules_bn_parameters_weight_
        l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_parameters_bias_ = L_self_modules_features_modules_20_modules_branch1_1b_modules_bn_parameters_bias_
        l_self_modules_features_modules_20_modules_branch2_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_20_modules_branch2_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_20_modules_branch2_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_20_modules_branch2_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_20_modules_branch2_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_20_modules_branch2_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_20_modules_branch2_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_20_modules_branch2_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_20_modules_branch2_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_20_modules_branch2_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_20_modules_branch2_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_20_modules_branch2_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_20_modules_branch2_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_20_modules_branch2_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_20_modules_branch2_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_20_modules_branch2_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_20_modules_branch2_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_20_modules_branch2_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_20_modules_branch2_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_20_modules_branch2_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_20_modules_branch2_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_20_modules_branch2_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_20_modules_branch2_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_20_modules_branch2_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_20_modules_branch2_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_20_modules_branch2_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_20_modules_branch2_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_20_modules_branch2_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_20_modules_branch2_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_20_modules_branch2_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_20_modules_branch2_3a_modules_conv_parameters_weight_ = L_self_modules_features_modules_20_modules_branch2_3a_modules_conv_parameters_weight_
        l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_20_modules_branch2_3a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_20_modules_branch2_3a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_parameters_weight_ = L_self_modules_features_modules_20_modules_branch2_3a_modules_bn_parameters_weight_
        l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_parameters_bias_ = L_self_modules_features_modules_20_modules_branch2_3a_modules_bn_parameters_bias_
        l_self_modules_features_modules_20_modules_branch2_3b_modules_conv_parameters_weight_ = L_self_modules_features_modules_20_modules_branch2_3b_modules_conv_parameters_weight_
        l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_20_modules_branch2_3b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_20_modules_branch2_3b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_parameters_weight_ = L_self_modules_features_modules_20_modules_branch2_3b_modules_bn_parameters_weight_
        l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_parameters_bias_ = L_self_modules_features_modules_20_modules_branch2_3b_modules_bn_parameters_bias_
        l_self_modules_features_modules_20_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_20_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_21_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_features_modules_21_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_features_modules_21_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_21_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_21_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_21_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_21_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_features_modules_21_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_features_modules_21_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_features_modules_21_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_features_modules_21_modules_branch1_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_21_modules_branch1_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_21_modules_branch1_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_21_modules_branch1_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_21_modules_branch1_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_21_modules_branch1_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_21_modules_branch1_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_21_modules_branch1_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_21_modules_branch1_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_21_modules_branch1_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_21_modules_branch1_1a_modules_conv_parameters_weight_ = L_self_modules_features_modules_21_modules_branch1_1a_modules_conv_parameters_weight_
        l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_21_modules_branch1_1a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_21_modules_branch1_1a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_parameters_weight_ = L_self_modules_features_modules_21_modules_branch1_1a_modules_bn_parameters_weight_
        l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_parameters_bias_ = L_self_modules_features_modules_21_modules_branch1_1a_modules_bn_parameters_bias_
        l_self_modules_features_modules_21_modules_branch1_1b_modules_conv_parameters_weight_ = L_self_modules_features_modules_21_modules_branch1_1b_modules_conv_parameters_weight_
        l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_21_modules_branch1_1b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_21_modules_branch1_1b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_parameters_weight_ = L_self_modules_features_modules_21_modules_branch1_1b_modules_bn_parameters_weight_
        l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_parameters_bias_ = L_self_modules_features_modules_21_modules_branch1_1b_modules_bn_parameters_bias_
        l_self_modules_features_modules_21_modules_branch2_0_modules_conv_parameters_weight_ = L_self_modules_features_modules_21_modules_branch2_0_modules_conv_parameters_weight_
        l_self_modules_features_modules_21_modules_branch2_0_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_21_modules_branch2_0_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_21_modules_branch2_0_modules_bn_buffers_running_var_ = L_self_modules_features_modules_21_modules_branch2_0_modules_bn_buffers_running_var_
        l_self_modules_features_modules_21_modules_branch2_0_modules_bn_parameters_weight_ = L_self_modules_features_modules_21_modules_branch2_0_modules_bn_parameters_weight_
        l_self_modules_features_modules_21_modules_branch2_0_modules_bn_parameters_bias_ = L_self_modules_features_modules_21_modules_branch2_0_modules_bn_parameters_bias_
        l_self_modules_features_modules_21_modules_branch2_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_21_modules_branch2_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_21_modules_branch2_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_21_modules_branch2_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_21_modules_branch2_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_21_modules_branch2_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_21_modules_branch2_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_21_modules_branch2_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_21_modules_branch2_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_21_modules_branch2_1_modules_bn_parameters_bias_
        l_self_modules_features_modules_21_modules_branch2_2_modules_conv_parameters_weight_ = L_self_modules_features_modules_21_modules_branch2_2_modules_conv_parameters_weight_
        l_self_modules_features_modules_21_modules_branch2_2_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_21_modules_branch2_2_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_21_modules_branch2_2_modules_bn_buffers_running_var_ = L_self_modules_features_modules_21_modules_branch2_2_modules_bn_buffers_running_var_
        l_self_modules_features_modules_21_modules_branch2_2_modules_bn_parameters_weight_ = L_self_modules_features_modules_21_modules_branch2_2_modules_bn_parameters_weight_
        l_self_modules_features_modules_21_modules_branch2_2_modules_bn_parameters_bias_ = L_self_modules_features_modules_21_modules_branch2_2_modules_bn_parameters_bias_
        l_self_modules_features_modules_21_modules_branch2_3a_modules_conv_parameters_weight_ = L_self_modules_features_modules_21_modules_branch2_3a_modules_conv_parameters_weight_
        l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_21_modules_branch2_3a_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_buffers_running_var_ = L_self_modules_features_modules_21_modules_branch2_3a_modules_bn_buffers_running_var_
        l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_parameters_weight_ = L_self_modules_features_modules_21_modules_branch2_3a_modules_bn_parameters_weight_
        l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_parameters_bias_ = L_self_modules_features_modules_21_modules_branch2_3a_modules_bn_parameters_bias_
        l_self_modules_features_modules_21_modules_branch2_3b_modules_conv_parameters_weight_ = L_self_modules_features_modules_21_modules_branch2_3b_modules_conv_parameters_weight_
        l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_21_modules_branch2_3b_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_buffers_running_var_ = L_self_modules_features_modules_21_modules_branch2_3b_modules_bn_buffers_running_var_
        l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_parameters_weight_ = L_self_modules_features_modules_21_modules_branch2_3b_modules_bn_parameters_weight_
        l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_parameters_bias_ = L_self_modules_features_modules_21_modules_branch2_3b_modules_bn_parameters_bias_
        l_self_modules_features_modules_21_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_features_modules_21_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_last_linear_parameters_weight_ = (
            L_self_modules_last_linear_parameters_weight_
        )
        l_self_modules_last_linear_parameters_bias_ = (
            L_self_modules_last_linear_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_features_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_features_modules_0_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_features_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x = (
            l_self_modules_features_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_features_modules_0_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_features_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_features_modules_1_modules_conv_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_features_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_3 = (
            l_self_modules_features_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_1_modules_bn_parameters_weight_
        ) = l_self_modules_features_modules_1_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_features_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_features_modules_2_modules_conv_parameters_weight_ = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_features_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_6 = (
            l_self_modules_features_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_2_modules_bn_parameters_weight_
        ) = l_self_modules_features_modules_2_modules_bn_parameters_bias_ = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x0 = torch.nn.functional.max_pool2d(
            x_8, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_features_modules_3_modules_conv_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_features_modules_3_modules_conv_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_features_modules_3_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_9 = l_self_modules_features_modules_3_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv_modules_bn_buffers_running_var_ = (
            l_self_modules_features_modules_3_modules_conv_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_conv_modules_bn_parameters_bias_
        ) = None
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        out = torch.cat((x0, x_11), 1)
        x0 = x_11 = None
        x_12 = torch.conv2d(
            out,
            l_self_modules_features_modules_4_modules_branch0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_branch0_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_12 = l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_4_modules_branch0_modules_0_modules_bn_parameters_bias_ = (None)
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_features_modules_4_modules_branch0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_features_modules_4_modules_branch0_modules_1_modules_conv_parameters_weight_ = (None)
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_15 = l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_4_modules_branch0_modules_1_modules_bn_parameters_bias_ = (None)
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            out,
            l_self_modules_features_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out = l_self_modules_features_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_ = (None)
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_18 = l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_features_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_features_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_21 = l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_23 = torch.nn.functional.relu(x_22, inplace=True)
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_features_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_features_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_24 = l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_26 = torch.nn.functional.relu(x_25, inplace=True)
        x_25 = None
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_features_modules_4_modules_branch1_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_features_modules_4_modules_branch1_modules_3_modules_conv_parameters_weight_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_parameters_weight_,
            l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_27 = l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_buffers_running_var_ = l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_parameters_weight_ = l_self_modules_features_modules_4_modules_branch1_modules_3_modules_bn_parameters_bias_ = (None)
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        out_1 = torch.cat((x_17, x_29), 1)
        x_17 = x_29 = None
        x_30 = torch.conv2d(
            out_1,
            l_self_modules_features_modules_5_modules_conv_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_features_modules_5_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_30 = l_self_modules_features_modules_5_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv_modules_bn_buffers_running_var_ = (
            l_self_modules_features_modules_5_modules_conv_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_conv_modules_bn_parameters_bias_
        ) = None
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x1 = torch.nn.functional.max_pool2d(
            out_1, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        out_1 = None
        out_2 = torch.cat((x_32, x1), 1)
        x_32 = x1 = None
        x_33 = torch.conv2d(
            out_2,
            l_self_modules_features_modules_6_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_features_modules_6_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_6_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_6_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_33 = l_self_modules_features_modules_6_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_6_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_6_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_6_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            out_2,
            l_self_modules_features_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_36 = l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_features_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_features_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_39 = l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        x_42 = torch.conv2d(
            out_2,
            l_self_modules_features_modules_6_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_42 = l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_6_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_44 = torch.nn.functional.relu(x_43, inplace=True)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_features_modules_6_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_features_modules_6_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_45 = l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_6_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_47 = torch.nn.functional.relu(x_46, inplace=True)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_features_modules_6_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_features_modules_6_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_48 = l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_6_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_50 = torch.nn.functional.relu(x_49, inplace=True)
        x_49 = None
        input_1 = torch._C._nn.avg_pool2d(out_2, 3, 1, 1, False, False, None)
        out_2 = None
        x_51 = torch.conv2d(
            input_1,
            l_self_modules_features_modules_6_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_features_modules_6_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_51 = l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_6_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        out_3 = torch.cat((x_35, x_41, x_50, x_53), 1)
        x_35 = x_41 = x_50 = x_53 = None
        x_54 = torch.conv2d(
            out_3,
            l_self_modules_features_modules_7_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_7_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_features_modules_7_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_7_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_7_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_54 = l_self_modules_features_modules_7_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_7_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_7_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_7_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        x_57 = torch.conv2d(
            out_3,
            l_self_modules_features_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_57 = l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_59 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_features_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_features_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_60 = l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_62 = torch.nn.functional.relu(x_61, inplace=True)
        x_61 = None
        x_63 = torch.conv2d(
            out_3,
            l_self_modules_features_modules_7_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_7_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_63 = l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_7_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.relu(x_64, inplace=True)
        x_64 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_features_modules_7_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_features_modules_7_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_66 = l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_7_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_features_modules_7_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_features_modules_7_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_69 = l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_7_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        input_2 = torch._C._nn.avg_pool2d(out_3, 3, 1, 1, False, False, None)
        out_3 = None
        x_72 = torch.conv2d(
            input_2,
            l_self_modules_features_modules_7_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_features_modules_7_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_72 = l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_7_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        out_4 = torch.cat((x_56, x_62, x_71, x_74), 1)
        x_56 = x_62 = x_71 = x_74 = None
        x_75 = torch.conv2d(
            out_4,
            l_self_modules_features_modules_8_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_8_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_features_modules_8_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_8_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_8_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_75 = l_self_modules_features_modules_8_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_8_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_8_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_8_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_77 = torch.nn.functional.relu(x_76, inplace=True)
        x_76 = None
        x_78 = torch.conv2d(
            out_4,
            l_self_modules_features_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_78 = l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_80 = torch.nn.functional.relu(x_79, inplace=True)
        x_79 = None
        x_81 = torch.conv2d(
            x_80,
            l_self_modules_features_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_80 = l_self_modules_features_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_81 = l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_83 = torch.nn.functional.relu(x_82, inplace=True)
        x_82 = None
        x_84 = torch.conv2d(
            out_4,
            l_self_modules_features_modules_8_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_8_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_84 = l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_8_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_features_modules_8_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_features_modules_8_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_87 = l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_8_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_89 = torch.nn.functional.relu(x_88, inplace=True)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_features_modules_8_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_features_modules_8_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_90 = l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_8_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_92 = torch.nn.functional.relu(x_91, inplace=True)
        x_91 = None
        input_3 = torch._C._nn.avg_pool2d(out_4, 3, 1, 1, False, False, None)
        out_4 = None
        x_93 = torch.conv2d(
            input_3,
            l_self_modules_features_modules_8_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_features_modules_8_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_93 = l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_8_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_95 = torch.nn.functional.relu(x_94, inplace=True)
        x_94 = None
        out_5 = torch.cat((x_77, x_83, x_92, x_95), 1)
        x_77 = x_83 = x_92 = x_95 = None
        x_96 = torch.conv2d(
            out_5,
            l_self_modules_features_modules_9_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_9_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_features_modules_9_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_9_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_9_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_96 = l_self_modules_features_modules_9_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_9_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_9_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_9_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_98 = torch.nn.functional.relu(x_97, inplace=True)
        x_97 = None
        x_99 = torch.conv2d(
            out_5,
            l_self_modules_features_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_99 = l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_features_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_features_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_102 = l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_105 = torch.conv2d(
            out_5,
            l_self_modules_features_modules_9_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_9_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_105 = l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_9_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_107 = torch.nn.functional.relu(x_106, inplace=True)
        x_106 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_features_modules_9_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_features_modules_9_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_108 = l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_9_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_110 = torch.nn.functional.relu(x_109, inplace=True)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_features_modules_9_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_110 = l_self_modules_features_modules_9_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_111 = l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_9_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        input_4 = torch._C._nn.avg_pool2d(out_5, 3, 1, 1, False, False, None)
        out_5 = None
        x_114 = torch.conv2d(
            input_4,
            l_self_modules_features_modules_9_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_features_modules_9_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_114 = l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_9_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_116 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        out_6 = torch.cat((x_98, x_104, x_113, x_116), 1)
        x_98 = x_104 = x_113 = x_116 = None
        x_117 = torch.conv2d(
            out_6,
            l_self_modules_features_modules_10_modules_branch0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_10_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_features_modules_10_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_10_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_10_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_117 = l_self_modules_features_modules_10_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_10_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_10_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_10_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_119 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.conv2d(
            out_6,
            l_self_modules_features_modules_10_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_10_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_120 = l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_10_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_122 = torch.nn.functional.relu(x_121, inplace=True)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_features_modules_10_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_features_modules_10_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_123 = l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_10_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_125 = torch.nn.functional.relu(x_124, inplace=True)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_features_modules_10_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_features_modules_10_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_126 = l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_10_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_128 = torch.nn.functional.relu(x_127, inplace=True)
        x_127 = None
        x2 = torch.nn.functional.max_pool2d(
            out_6, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        out_6 = None
        out_7 = torch.cat((x_119, x_128, x2), 1)
        x_119 = x_128 = x2 = None
        x_129 = torch.conv2d(
            out_7,
            l_self_modules_features_modules_11_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_11_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_features_modules_11_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_129 = l_self_modules_features_modules_11_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_131 = torch.nn.functional.relu(x_130, inplace=True)
        x_130 = None
        x_132 = torch.conv2d(
            out_7,
            l_self_modules_features_modules_11_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_11_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_132 = l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_features_modules_11_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_features_modules_11_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_135 = l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_137 = torch.nn.functional.relu(x_136, inplace=True)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_features_modules_11_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_features_modules_11_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_138 = l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_140 = torch.nn.functional.relu(x_139, inplace=True)
        x_139 = None
        x_141 = torch.conv2d(
            out_7,
            l_self_modules_features_modules_11_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_11_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_141 = l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_143 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_features_modules_11_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_features_modules_11_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_144 = l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_146 = torch.nn.functional.relu(x_145, inplace=True)
        x_145 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_features_modules_11_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_features_modules_11_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_147 = l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_149 = torch.nn.functional.relu(x_148, inplace=True)
        x_148 = None
        x_150 = torch.conv2d(
            x_149,
            l_self_modules_features_modules_11_modules_branch2_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_149 = l_self_modules_features_modules_11_modules_branch2_modules_3_modules_conv_parameters_weight_ = (None)
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_150 = l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_branch2_modules_3_modules_bn_parameters_bias_ = (None)
        x_152 = torch.nn.functional.relu(x_151, inplace=True)
        x_151 = None
        x_153 = torch.conv2d(
            x_152,
            l_self_modules_features_modules_11_modules_branch2_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_features_modules_11_modules_branch2_modules_4_modules_conv_parameters_weight_ = (None)
        x_154 = torch.nn.functional.batch_norm(
            x_153,
            l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_153 = l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_branch2_modules_4_modules_bn_parameters_bias_ = (None)
        x_155 = torch.nn.functional.relu(x_154, inplace=True)
        x_154 = None
        input_5 = torch._C._nn.avg_pool2d(out_7, 3, 1, 1, False, False, None)
        out_7 = None
        x_156 = torch.conv2d(
            input_5,
            l_self_modules_features_modules_11_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_features_modules_11_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_156 = l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_158 = torch.nn.functional.relu(x_157, inplace=True)
        x_157 = None
        out_8 = torch.cat((x_131, x_140, x_155, x_158), 1)
        x_131 = x_140 = x_155 = x_158 = None
        x_159 = torch.conv2d(
            out_8,
            l_self_modules_features_modules_12_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_12_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_features_modules_12_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_159 = l_self_modules_features_modules_12_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_161 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        x_162 = torch.conv2d(
            out_8,
            l_self_modules_features_modules_12_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_12_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_162 = l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_164 = torch.nn.functional.relu(x_163, inplace=True)
        x_163 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_features_modules_12_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_164 = l_self_modules_features_modules_12_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_165 = l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_167 = torch.nn.functional.relu(x_166, inplace=True)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_features_modules_12_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_features_modules_12_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_168 = l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_170 = torch.nn.functional.relu(x_169, inplace=True)
        x_169 = None
        x_171 = torch.conv2d(
            out_8,
            l_self_modules_features_modules_12_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_12_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_172 = torch.nn.functional.batch_norm(
            x_171,
            l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_171 = l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_173 = torch.nn.functional.relu(x_172, inplace=True)
        x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_features_modules_12_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_173 = l_self_modules_features_modules_12_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_174 = l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_176 = torch.nn.functional.relu(x_175, inplace=True)
        x_175 = None
        x_177 = torch.conv2d(
            x_176,
            l_self_modules_features_modules_12_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_176 = l_self_modules_features_modules_12_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_177 = l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_179 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_features_modules_12_modules_branch2_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_features_modules_12_modules_branch2_modules_3_modules_conv_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_180 = l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_branch2_modules_3_modules_bn_parameters_bias_ = (None)
        x_182 = torch.nn.functional.relu(x_181, inplace=True)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_features_modules_12_modules_branch2_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_features_modules_12_modules_branch2_modules_4_modules_conv_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_183 = l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_branch2_modules_4_modules_bn_parameters_bias_ = (None)
        x_185 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        input_6 = torch._C._nn.avg_pool2d(out_8, 3, 1, 1, False, False, None)
        out_8 = None
        x_186 = torch.conv2d(
            input_6,
            l_self_modules_features_modules_12_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_features_modules_12_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_186 = l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_188 = torch.nn.functional.relu(x_187, inplace=True)
        x_187 = None
        out_9 = torch.cat((x_161, x_170, x_185, x_188), 1)
        x_161 = x_170 = x_185 = x_188 = None
        x_189 = torch.conv2d(
            out_9,
            l_self_modules_features_modules_13_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_13_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_features_modules_13_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_189 = l_self_modules_features_modules_13_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_191 = torch.nn.functional.relu(x_190, inplace=True)
        x_190 = None
        x_192 = torch.conv2d(
            out_9,
            l_self_modules_features_modules_13_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_13_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_192 = l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_194 = torch.nn.functional.relu(x_193, inplace=True)
        x_193 = None
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_features_modules_13_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_194 = l_self_modules_features_modules_13_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_196 = torch.nn.functional.batch_norm(
            x_195,
            l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_195 = l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_197 = torch.nn.functional.relu(x_196, inplace=True)
        x_196 = None
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_features_modules_13_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_features_modules_13_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_198 = l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_200 = torch.nn.functional.relu(x_199, inplace=True)
        x_199 = None
        x_201 = torch.conv2d(
            out_9,
            l_self_modules_features_modules_13_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_13_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_201 = l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_203 = torch.nn.functional.relu(x_202, inplace=True)
        x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_features_modules_13_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_203 = l_self_modules_features_modules_13_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_204 = l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_206 = torch.nn.functional.relu(x_205, inplace=True)
        x_205 = None
        x_207 = torch.conv2d(
            x_206,
            l_self_modules_features_modules_13_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_206 = l_self_modules_features_modules_13_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_207 = l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_209 = torch.nn.functional.relu(x_208, inplace=True)
        x_208 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_features_modules_13_modules_branch2_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_features_modules_13_modules_branch2_modules_3_modules_conv_parameters_weight_ = (None)
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_210 = l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_branch2_modules_3_modules_bn_parameters_bias_ = (None)
        x_212 = torch.nn.functional.relu(x_211, inplace=True)
        x_211 = None
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_features_modules_13_modules_branch2_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_212 = l_self_modules_features_modules_13_modules_branch2_modules_4_modules_conv_parameters_weight_ = (None)
        x_214 = torch.nn.functional.batch_norm(
            x_213,
            l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_213 = l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_branch2_modules_4_modules_bn_parameters_bias_ = (None)
        x_215 = torch.nn.functional.relu(x_214, inplace=True)
        x_214 = None
        input_7 = torch._C._nn.avg_pool2d(out_9, 3, 1, 1, False, False, None)
        out_9 = None
        x_216 = torch.conv2d(
            input_7,
            l_self_modules_features_modules_13_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_features_modules_13_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_216 = l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_218 = torch.nn.functional.relu(x_217, inplace=True)
        x_217 = None
        out_10 = torch.cat((x_191, x_200, x_215, x_218), 1)
        x_191 = x_200 = x_215 = x_218 = None
        x_219 = torch.conv2d(
            out_10,
            l_self_modules_features_modules_14_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_14_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_220 = torch.nn.functional.batch_norm(
            x_219,
            l_self_modules_features_modules_14_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_219 = l_self_modules_features_modules_14_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_221 = torch.nn.functional.relu(x_220, inplace=True)
        x_220 = None
        x_222 = torch.conv2d(
            out_10,
            l_self_modules_features_modules_14_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_14_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_223 = torch.nn.functional.batch_norm(
            x_222,
            l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_222 = l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_224 = torch.nn.functional.relu(x_223, inplace=True)
        x_223 = None
        x_225 = torch.conv2d(
            x_224,
            l_self_modules_features_modules_14_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_224 = l_self_modules_features_modules_14_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_226 = torch.nn.functional.batch_norm(
            x_225,
            l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_225 = l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_227 = torch.nn.functional.relu(x_226, inplace=True)
        x_226 = None
        x_228 = torch.conv2d(
            x_227,
            l_self_modules_features_modules_14_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_227 = l_self_modules_features_modules_14_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_229 = torch.nn.functional.batch_norm(
            x_228,
            l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_228 = l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_230 = torch.nn.functional.relu(x_229, inplace=True)
        x_229 = None
        x_231 = torch.conv2d(
            out_10,
            l_self_modules_features_modules_14_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_14_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_232 = torch.nn.functional.batch_norm(
            x_231,
            l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_231 = l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_233 = torch.nn.functional.relu(x_232, inplace=True)
        x_232 = None
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_features_modules_14_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_233 = l_self_modules_features_modules_14_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_234 = l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_236 = torch.nn.functional.relu(x_235, inplace=True)
        x_235 = None
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_features_modules_14_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_236 = l_self_modules_features_modules_14_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_238 = torch.nn.functional.batch_norm(
            x_237,
            l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_237 = l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_239 = torch.nn.functional.relu(x_238, inplace=True)
        x_238 = None
        x_240 = torch.conv2d(
            x_239,
            l_self_modules_features_modules_14_modules_branch2_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_239 = l_self_modules_features_modules_14_modules_branch2_modules_3_modules_conv_parameters_weight_ = (None)
        x_241 = torch.nn.functional.batch_norm(
            x_240,
            l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_240 = l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_branch2_modules_3_modules_bn_parameters_bias_ = (None)
        x_242 = torch.nn.functional.relu(x_241, inplace=True)
        x_241 = None
        x_243 = torch.conv2d(
            x_242,
            l_self_modules_features_modules_14_modules_branch2_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_242 = l_self_modules_features_modules_14_modules_branch2_modules_4_modules_conv_parameters_weight_ = (None)
        x_244 = torch.nn.functional.batch_norm(
            x_243,
            l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_243 = l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_branch2_modules_4_modules_bn_parameters_bias_ = (None)
        x_245 = torch.nn.functional.relu(x_244, inplace=True)
        x_244 = None
        input_8 = torch._C._nn.avg_pool2d(out_10, 3, 1, 1, False, False, None)
        out_10 = None
        x_246 = torch.conv2d(
            input_8,
            l_self_modules_features_modules_14_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_8 = l_self_modules_features_modules_14_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_247 = torch.nn.functional.batch_norm(
            x_246,
            l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_246 = l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_248 = torch.nn.functional.relu(x_247, inplace=True)
        x_247 = None
        out_11 = torch.cat((x_221, x_230, x_245, x_248), 1)
        x_221 = x_230 = x_245 = x_248 = None
        x_249 = torch.conv2d(
            out_11,
            l_self_modules_features_modules_15_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_15_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_250 = torch.nn.functional.batch_norm(
            x_249,
            l_self_modules_features_modules_15_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_249 = l_self_modules_features_modules_15_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_251 = torch.nn.functional.relu(x_250, inplace=True)
        x_250 = None
        x_252 = torch.conv2d(
            out_11,
            l_self_modules_features_modules_15_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_15_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_253 = torch.nn.functional.batch_norm(
            x_252,
            l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_252 = l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_254 = torch.nn.functional.relu(x_253, inplace=True)
        x_253 = None
        x_255 = torch.conv2d(
            x_254,
            l_self_modules_features_modules_15_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_254 = l_self_modules_features_modules_15_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_256 = torch.nn.functional.batch_norm(
            x_255,
            l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_255 = l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_257 = torch.nn.functional.relu(x_256, inplace=True)
        x_256 = None
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_features_modules_15_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_features_modules_15_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_259 = torch.nn.functional.batch_norm(
            x_258,
            l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_258 = l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_260 = torch.nn.functional.relu(x_259, inplace=True)
        x_259 = None
        x_261 = torch.conv2d(
            out_11,
            l_self_modules_features_modules_15_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_15_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_262 = torch.nn.functional.batch_norm(
            x_261,
            l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_261 = l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_263 = torch.nn.functional.relu(x_262, inplace=True)
        x_262 = None
        x_264 = torch.conv2d(
            x_263,
            l_self_modules_features_modules_15_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_263 = l_self_modules_features_modules_15_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_265 = torch.nn.functional.batch_norm(
            x_264,
            l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_264 = l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_266 = torch.nn.functional.relu(x_265, inplace=True)
        x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_features_modules_15_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_266 = l_self_modules_features_modules_15_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_267 = l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_269 = torch.nn.functional.relu(x_268, inplace=True)
        x_268 = None
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_features_modules_15_modules_branch2_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_269 = l_self_modules_features_modules_15_modules_branch2_modules_3_modules_conv_parameters_weight_ = (None)
        x_271 = torch.nn.functional.batch_norm(
            x_270,
            l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_270 = l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_branch2_modules_3_modules_bn_parameters_bias_ = (None)
        x_272 = torch.nn.functional.relu(x_271, inplace=True)
        x_271 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_features_modules_15_modules_branch2_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_272 = l_self_modules_features_modules_15_modules_branch2_modules_4_modules_conv_parameters_weight_ = (None)
        x_274 = torch.nn.functional.batch_norm(
            x_273,
            l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_273 = l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_branch2_modules_4_modules_bn_parameters_bias_ = (None)
        x_275 = torch.nn.functional.relu(x_274, inplace=True)
        x_274 = None
        input_9 = torch._C._nn.avg_pool2d(out_11, 3, 1, 1, False, False, None)
        out_11 = None
        x_276 = torch.conv2d(
            input_9,
            l_self_modules_features_modules_15_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_9 = l_self_modules_features_modules_15_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_277 = torch.nn.functional.batch_norm(
            x_276,
            l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_276 = l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_278 = torch.nn.functional.relu(x_277, inplace=True)
        x_277 = None
        out_12 = torch.cat((x_251, x_260, x_275, x_278), 1)
        x_251 = x_260 = x_275 = x_278 = None
        x_279 = torch.conv2d(
            out_12,
            l_self_modules_features_modules_16_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_16_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_280 = torch.nn.functional.batch_norm(
            x_279,
            l_self_modules_features_modules_16_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_16_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_16_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_279 = l_self_modules_features_modules_16_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_16_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_16_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_16_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_281 = torch.nn.functional.relu(x_280, inplace=True)
        x_280 = None
        x_282 = torch.conv2d(
            out_12,
            l_self_modules_features_modules_16_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_16_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_283 = torch.nn.functional.batch_norm(
            x_282,
            l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_282 = l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_16_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_284 = torch.nn.functional.relu(x_283, inplace=True)
        x_283 = None
        x_285 = torch.conv2d(
            x_284,
            l_self_modules_features_modules_16_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_284 = l_self_modules_features_modules_16_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_286 = torch.nn.functional.batch_norm(
            x_285,
            l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_285 = l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_16_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_287 = torch.nn.functional.relu(x_286, inplace=True)
        x_286 = None
        x_288 = torch.conv2d(
            x_287,
            l_self_modules_features_modules_16_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_287 = l_self_modules_features_modules_16_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_289 = torch.nn.functional.batch_norm(
            x_288,
            l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_288 = l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_16_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_290 = torch.nn.functional.relu(x_289, inplace=True)
        x_289 = None
        x_291 = torch.conv2d(
            out_12,
            l_self_modules_features_modules_16_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_16_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_292 = torch.nn.functional.batch_norm(
            x_291,
            l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_291 = l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_16_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_293 = torch.nn.functional.relu(x_292, inplace=True)
        x_292 = None
        x_294 = torch.conv2d(
            x_293,
            l_self_modules_features_modules_16_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_293 = l_self_modules_features_modules_16_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_295 = torch.nn.functional.batch_norm(
            x_294,
            l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_294 = l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_16_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_296 = torch.nn.functional.relu(x_295, inplace=True)
        x_295 = None
        x_297 = torch.conv2d(
            x_296,
            l_self_modules_features_modules_16_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_296 = l_self_modules_features_modules_16_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_298 = torch.nn.functional.batch_norm(
            x_297,
            l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_297 = l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_16_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_299 = torch.nn.functional.relu(x_298, inplace=True)
        x_298 = None
        x_300 = torch.conv2d(
            x_299,
            l_self_modules_features_modules_16_modules_branch2_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_299 = l_self_modules_features_modules_16_modules_branch2_modules_3_modules_conv_parameters_weight_ = (None)
        x_301 = torch.nn.functional.batch_norm(
            x_300,
            l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_parameters_weight_,
            l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_300 = l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_buffers_running_var_ = l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_parameters_weight_ = l_self_modules_features_modules_16_modules_branch2_modules_3_modules_bn_parameters_bias_ = (None)
        x_302 = torch.nn.functional.relu(x_301, inplace=True)
        x_301 = None
        x_303 = torch.conv2d(
            x_302,
            l_self_modules_features_modules_16_modules_branch2_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_302 = l_self_modules_features_modules_16_modules_branch2_modules_4_modules_conv_parameters_weight_ = (None)
        x_304 = torch.nn.functional.batch_norm(
            x_303,
            l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_parameters_weight_,
            l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_303 = l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_buffers_running_var_ = l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_parameters_weight_ = l_self_modules_features_modules_16_modules_branch2_modules_4_modules_bn_parameters_bias_ = (None)
        x_305 = torch.nn.functional.relu(x_304, inplace=True)
        x_304 = None
        input_10 = torch._C._nn.avg_pool2d(out_12, 3, 1, 1, False, False, None)
        out_12 = None
        x_306 = torch.conv2d(
            input_10,
            l_self_modules_features_modules_16_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_10 = l_self_modules_features_modules_16_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_307 = torch.nn.functional.batch_norm(
            x_306,
            l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_306 = l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_16_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_308 = torch.nn.functional.relu(x_307, inplace=True)
        x_307 = None
        out_13 = torch.cat((x_281, x_290, x_305, x_308), 1)
        x_281 = x_290 = x_305 = x_308 = None
        x_309 = torch.conv2d(
            out_13,
            l_self_modules_features_modules_17_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_17_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_310 = torch.nn.functional.batch_norm(
            x_309,
            l_self_modules_features_modules_17_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_17_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_17_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_309 = l_self_modules_features_modules_17_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_17_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_17_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_17_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_311 = torch.nn.functional.relu(x_310, inplace=True)
        x_310 = None
        x_312 = torch.conv2d(
            out_13,
            l_self_modules_features_modules_17_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_17_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_313 = torch.nn.functional.batch_norm(
            x_312,
            l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_312 = l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_17_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_314 = torch.nn.functional.relu(x_313, inplace=True)
        x_313 = None
        x_315 = torch.conv2d(
            x_314,
            l_self_modules_features_modules_17_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_314 = l_self_modules_features_modules_17_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_316 = torch.nn.functional.batch_norm(
            x_315,
            l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_315 = l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_17_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_317 = torch.nn.functional.relu(x_316, inplace=True)
        x_316 = None
        x_318 = torch.conv2d(
            x_317,
            l_self_modules_features_modules_17_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_317 = l_self_modules_features_modules_17_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_319 = torch.nn.functional.batch_norm(
            x_318,
            l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_318 = l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_17_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_320 = torch.nn.functional.relu(x_319, inplace=True)
        x_319 = None
        x_321 = torch.conv2d(
            out_13,
            l_self_modules_features_modules_17_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_17_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_322 = torch.nn.functional.batch_norm(
            x_321,
            l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_321 = l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_17_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_323 = torch.nn.functional.relu(x_322, inplace=True)
        x_322 = None
        x_324 = torch.conv2d(
            x_323,
            l_self_modules_features_modules_17_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_323 = l_self_modules_features_modules_17_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_325 = torch.nn.functional.batch_norm(
            x_324,
            l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_324 = l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_17_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_326 = torch.nn.functional.relu(x_325, inplace=True)
        x_325 = None
        x_327 = torch.conv2d(
            x_326,
            l_self_modules_features_modules_17_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_326 = l_self_modules_features_modules_17_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_328 = torch.nn.functional.batch_norm(
            x_327,
            l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_327 = l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_17_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_329 = torch.nn.functional.relu(x_328, inplace=True)
        x_328 = None
        x_330 = torch.conv2d(
            x_329,
            l_self_modules_features_modules_17_modules_branch2_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_329 = l_self_modules_features_modules_17_modules_branch2_modules_3_modules_conv_parameters_weight_ = (None)
        x_331 = torch.nn.functional.batch_norm(
            x_330,
            l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_parameters_weight_,
            l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_330 = l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_buffers_running_var_ = l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_parameters_weight_ = l_self_modules_features_modules_17_modules_branch2_modules_3_modules_bn_parameters_bias_ = (None)
        x_332 = torch.nn.functional.relu(x_331, inplace=True)
        x_331 = None
        x_333 = torch.conv2d(
            x_332,
            l_self_modules_features_modules_17_modules_branch2_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_332 = l_self_modules_features_modules_17_modules_branch2_modules_4_modules_conv_parameters_weight_ = (None)
        x_334 = torch.nn.functional.batch_norm(
            x_333,
            l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_parameters_weight_,
            l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_333 = l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_buffers_running_var_ = l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_parameters_weight_ = l_self_modules_features_modules_17_modules_branch2_modules_4_modules_bn_parameters_bias_ = (None)
        x_335 = torch.nn.functional.relu(x_334, inplace=True)
        x_334 = None
        input_11 = torch._C._nn.avg_pool2d(out_13, 3, 1, 1, False, False, None)
        out_13 = None
        x_336 = torch.conv2d(
            input_11,
            l_self_modules_features_modules_17_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_11 = l_self_modules_features_modules_17_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_337 = torch.nn.functional.batch_norm(
            x_336,
            l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_336 = l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_17_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_338 = torch.nn.functional.relu(x_337, inplace=True)
        x_337 = None
        out_14 = torch.cat((x_311, x_320, x_335, x_338), 1)
        x_311 = x_320 = x_335 = x_338 = None
        x_339 = torch.conv2d(
            out_14,
            l_self_modules_features_modules_18_modules_branch0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_18_modules_branch0_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_340 = torch.nn.functional.batch_norm(
            x_339,
            l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_339 = l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_18_modules_branch0_modules_0_modules_bn_parameters_bias_ = (None)
        x_341 = torch.nn.functional.relu(x_340, inplace=True)
        x_340 = None
        x_342 = torch.conv2d(
            x_341,
            l_self_modules_features_modules_18_modules_branch0_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_341 = l_self_modules_features_modules_18_modules_branch0_modules_1_modules_conv_parameters_weight_ = (None)
        x_343 = torch.nn.functional.batch_norm(
            x_342,
            l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_342 = l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_18_modules_branch0_modules_1_modules_bn_parameters_bias_ = (None)
        x_344 = torch.nn.functional.relu(x_343, inplace=True)
        x_343 = None
        x_345 = torch.conv2d(
            out_14,
            l_self_modules_features_modules_18_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_18_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_346 = torch.nn.functional.batch_norm(
            x_345,
            l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_345 = l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_18_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_347 = torch.nn.functional.relu(x_346, inplace=True)
        x_346 = None
        x_348 = torch.conv2d(
            x_347,
            l_self_modules_features_modules_18_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_347 = l_self_modules_features_modules_18_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_349 = torch.nn.functional.batch_norm(
            x_348,
            l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_348 = l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_18_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_350 = torch.nn.functional.relu(x_349, inplace=True)
        x_349 = None
        x_351 = torch.conv2d(
            x_350,
            l_self_modules_features_modules_18_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_350 = l_self_modules_features_modules_18_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_352 = torch.nn.functional.batch_norm(
            x_351,
            l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_351 = l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_18_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_353 = torch.nn.functional.relu(x_352, inplace=True)
        x_352 = None
        x_354 = torch.conv2d(
            x_353,
            l_self_modules_features_modules_18_modules_branch1_modules_3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_353 = l_self_modules_features_modules_18_modules_branch1_modules_3_modules_conv_parameters_weight_ = (None)
        x_355 = torch.nn.functional.batch_norm(
            x_354,
            l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_parameters_weight_,
            l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_354 = l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_buffers_running_var_ = l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_parameters_weight_ = l_self_modules_features_modules_18_modules_branch1_modules_3_modules_bn_parameters_bias_ = (None)
        x_356 = torch.nn.functional.relu(x_355, inplace=True)
        x_355 = None
        x2_1 = torch.nn.functional.max_pool2d(
            out_14, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        out_14 = None
        out_15 = torch.cat((x_344, x_356, x2_1), 1)
        x_344 = x_356 = x2_1 = None
        x_357 = torch.conv2d(
            out_15,
            l_self_modules_features_modules_19_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_19_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_358 = torch.nn.functional.batch_norm(
            x_357,
            l_self_modules_features_modules_19_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_19_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_19_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_19_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_357 = l_self_modules_features_modules_19_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_19_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_19_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_19_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_359 = torch.nn.functional.relu(x_358, inplace=True)
        x_358 = None
        x_360 = torch.conv2d(
            out_15,
            l_self_modules_features_modules_19_modules_branch1_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_19_modules_branch1_0_modules_conv_parameters_weight_ = (
            None
        )
        x_361 = torch.nn.functional.batch_norm(
            x_360,
            l_self_modules_features_modules_19_modules_branch1_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_19_modules_branch1_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_19_modules_branch1_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_19_modules_branch1_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_360 = l_self_modules_features_modules_19_modules_branch1_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_19_modules_branch1_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_19_modules_branch1_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_19_modules_branch1_0_modules_bn_parameters_bias_ = (None)
        x_362 = torch.nn.functional.relu(x_361, inplace=True)
        x_361 = None
        x_363 = torch.conv2d(
            x_362,
            l_self_modules_features_modules_19_modules_branch1_1a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_19_modules_branch1_1a_modules_conv_parameters_weight_ = (
            None
        )
        x_364 = torch.nn.functional.batch_norm(
            x_363,
            l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_363 = l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_parameters_weight_ = l_self_modules_features_modules_19_modules_branch1_1a_modules_bn_parameters_bias_ = (None)
        x_365 = torch.nn.functional.relu(x_364, inplace=True)
        x_364 = None
        x_366 = torch.conv2d(
            x_362,
            l_self_modules_features_modules_19_modules_branch1_1b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_362 = l_self_modules_features_modules_19_modules_branch1_1b_modules_conv_parameters_weight_ = (None)
        x_367 = torch.nn.functional.batch_norm(
            x_366,
            l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_366 = l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_parameters_weight_ = l_self_modules_features_modules_19_modules_branch1_1b_modules_bn_parameters_bias_ = (None)
        x_368 = torch.nn.functional.relu(x_367, inplace=True)
        x_367 = None
        x1_1 = torch.cat((x_365, x_368), 1)
        x_365 = x_368 = None
        x_369 = torch.conv2d(
            out_15,
            l_self_modules_features_modules_19_modules_branch2_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_19_modules_branch2_0_modules_conv_parameters_weight_ = (
            None
        )
        x_370 = torch.nn.functional.batch_norm(
            x_369,
            l_self_modules_features_modules_19_modules_branch2_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_19_modules_branch2_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_19_modules_branch2_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_19_modules_branch2_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_369 = l_self_modules_features_modules_19_modules_branch2_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_19_modules_branch2_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_19_modules_branch2_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_19_modules_branch2_0_modules_bn_parameters_bias_ = (None)
        x_371 = torch.nn.functional.relu(x_370, inplace=True)
        x_370 = None
        x_372 = torch.conv2d(
            x_371,
            l_self_modules_features_modules_19_modules_branch2_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_371 = l_self_modules_features_modules_19_modules_branch2_1_modules_conv_parameters_weight_ = (None)
        x_373 = torch.nn.functional.batch_norm(
            x_372,
            l_self_modules_features_modules_19_modules_branch2_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_19_modules_branch2_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_19_modules_branch2_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_19_modules_branch2_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_372 = l_self_modules_features_modules_19_modules_branch2_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_19_modules_branch2_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_19_modules_branch2_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_19_modules_branch2_1_modules_bn_parameters_bias_ = (None)
        x_374 = torch.nn.functional.relu(x_373, inplace=True)
        x_373 = None
        x_375 = torch.conv2d(
            x_374,
            l_self_modules_features_modules_19_modules_branch2_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_374 = l_self_modules_features_modules_19_modules_branch2_2_modules_conv_parameters_weight_ = (None)
        x_376 = torch.nn.functional.batch_norm(
            x_375,
            l_self_modules_features_modules_19_modules_branch2_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_19_modules_branch2_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_19_modules_branch2_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_19_modules_branch2_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_375 = l_self_modules_features_modules_19_modules_branch2_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_19_modules_branch2_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_19_modules_branch2_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_19_modules_branch2_2_modules_bn_parameters_bias_ = (None)
        x_377 = torch.nn.functional.relu(x_376, inplace=True)
        x_376 = None
        x_378 = torch.conv2d(
            x_377,
            l_self_modules_features_modules_19_modules_branch2_3a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_19_modules_branch2_3a_modules_conv_parameters_weight_ = (
            None
        )
        x_379 = torch.nn.functional.batch_norm(
            x_378,
            l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_378 = l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_parameters_weight_ = l_self_modules_features_modules_19_modules_branch2_3a_modules_bn_parameters_bias_ = (None)
        x_380 = torch.nn.functional.relu(x_379, inplace=True)
        x_379 = None
        x_381 = torch.conv2d(
            x_377,
            l_self_modules_features_modules_19_modules_branch2_3b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_377 = l_self_modules_features_modules_19_modules_branch2_3b_modules_conv_parameters_weight_ = (None)
        x_382 = torch.nn.functional.batch_norm(
            x_381,
            l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_381 = l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_parameters_weight_ = l_self_modules_features_modules_19_modules_branch2_3b_modules_bn_parameters_bias_ = (None)
        x_383 = torch.nn.functional.relu(x_382, inplace=True)
        x_382 = None
        x2_2 = torch.cat((x_380, x_383), 1)
        x_380 = x_383 = None
        input_12 = torch._C._nn.avg_pool2d(out_15, 3, 1, 1, False, False, None)
        out_15 = None
        x_384 = torch.conv2d(
            input_12,
            l_self_modules_features_modules_19_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_12 = l_self_modules_features_modules_19_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_385 = torch.nn.functional.batch_norm(
            x_384,
            l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_384 = l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_19_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_386 = torch.nn.functional.relu(x_385, inplace=True)
        x_385 = None
        out_16 = torch.cat((x_359, x1_1, x2_2, x_386), 1)
        x_359 = x1_1 = x2_2 = x_386 = None
        x_387 = torch.conv2d(
            out_16,
            l_self_modules_features_modules_20_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_20_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_388 = torch.nn.functional.batch_norm(
            x_387,
            l_self_modules_features_modules_20_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_20_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_20_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_20_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_387 = l_self_modules_features_modules_20_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_20_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_20_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_20_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_389 = torch.nn.functional.relu(x_388, inplace=True)
        x_388 = None
        x_390 = torch.conv2d(
            out_16,
            l_self_modules_features_modules_20_modules_branch1_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_20_modules_branch1_0_modules_conv_parameters_weight_ = (
            None
        )
        x_391 = torch.nn.functional.batch_norm(
            x_390,
            l_self_modules_features_modules_20_modules_branch1_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_20_modules_branch1_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_20_modules_branch1_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_20_modules_branch1_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_390 = l_self_modules_features_modules_20_modules_branch1_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_20_modules_branch1_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_20_modules_branch1_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_20_modules_branch1_0_modules_bn_parameters_bias_ = (None)
        x_392 = torch.nn.functional.relu(x_391, inplace=True)
        x_391 = None
        x_393 = torch.conv2d(
            x_392,
            l_self_modules_features_modules_20_modules_branch1_1a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_20_modules_branch1_1a_modules_conv_parameters_weight_ = (
            None
        )
        x_394 = torch.nn.functional.batch_norm(
            x_393,
            l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_393 = l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_parameters_weight_ = l_self_modules_features_modules_20_modules_branch1_1a_modules_bn_parameters_bias_ = (None)
        x_395 = torch.nn.functional.relu(x_394, inplace=True)
        x_394 = None
        x_396 = torch.conv2d(
            x_392,
            l_self_modules_features_modules_20_modules_branch1_1b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_392 = l_self_modules_features_modules_20_modules_branch1_1b_modules_conv_parameters_weight_ = (None)
        x_397 = torch.nn.functional.batch_norm(
            x_396,
            l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_396 = l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_parameters_weight_ = l_self_modules_features_modules_20_modules_branch1_1b_modules_bn_parameters_bias_ = (None)
        x_398 = torch.nn.functional.relu(x_397, inplace=True)
        x_397 = None
        x1_2 = torch.cat((x_395, x_398), 1)
        x_395 = x_398 = None
        x_399 = torch.conv2d(
            out_16,
            l_self_modules_features_modules_20_modules_branch2_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_20_modules_branch2_0_modules_conv_parameters_weight_ = (
            None
        )
        x_400 = torch.nn.functional.batch_norm(
            x_399,
            l_self_modules_features_modules_20_modules_branch2_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_20_modules_branch2_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_20_modules_branch2_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_20_modules_branch2_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_399 = l_self_modules_features_modules_20_modules_branch2_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_20_modules_branch2_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_20_modules_branch2_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_20_modules_branch2_0_modules_bn_parameters_bias_ = (None)
        x_401 = torch.nn.functional.relu(x_400, inplace=True)
        x_400 = None
        x_402 = torch.conv2d(
            x_401,
            l_self_modules_features_modules_20_modules_branch2_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_401 = l_self_modules_features_modules_20_modules_branch2_1_modules_conv_parameters_weight_ = (None)
        x_403 = torch.nn.functional.batch_norm(
            x_402,
            l_self_modules_features_modules_20_modules_branch2_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_20_modules_branch2_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_20_modules_branch2_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_20_modules_branch2_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_402 = l_self_modules_features_modules_20_modules_branch2_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_20_modules_branch2_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_20_modules_branch2_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_20_modules_branch2_1_modules_bn_parameters_bias_ = (None)
        x_404 = torch.nn.functional.relu(x_403, inplace=True)
        x_403 = None
        x_405 = torch.conv2d(
            x_404,
            l_self_modules_features_modules_20_modules_branch2_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_404 = l_self_modules_features_modules_20_modules_branch2_2_modules_conv_parameters_weight_ = (None)
        x_406 = torch.nn.functional.batch_norm(
            x_405,
            l_self_modules_features_modules_20_modules_branch2_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_20_modules_branch2_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_20_modules_branch2_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_20_modules_branch2_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_405 = l_self_modules_features_modules_20_modules_branch2_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_20_modules_branch2_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_20_modules_branch2_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_20_modules_branch2_2_modules_bn_parameters_bias_ = (None)
        x_407 = torch.nn.functional.relu(x_406, inplace=True)
        x_406 = None
        x_408 = torch.conv2d(
            x_407,
            l_self_modules_features_modules_20_modules_branch2_3a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_20_modules_branch2_3a_modules_conv_parameters_weight_ = (
            None
        )
        x_409 = torch.nn.functional.batch_norm(
            x_408,
            l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_408 = l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_parameters_weight_ = l_self_modules_features_modules_20_modules_branch2_3a_modules_bn_parameters_bias_ = (None)
        x_410 = torch.nn.functional.relu(x_409, inplace=True)
        x_409 = None
        x_411 = torch.conv2d(
            x_407,
            l_self_modules_features_modules_20_modules_branch2_3b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_407 = l_self_modules_features_modules_20_modules_branch2_3b_modules_conv_parameters_weight_ = (None)
        x_412 = torch.nn.functional.batch_norm(
            x_411,
            l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_411 = l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_parameters_weight_ = l_self_modules_features_modules_20_modules_branch2_3b_modules_bn_parameters_bias_ = (None)
        x_413 = torch.nn.functional.relu(x_412, inplace=True)
        x_412 = None
        x2_3 = torch.cat((x_410, x_413), 1)
        x_410 = x_413 = None
        input_13 = torch._C._nn.avg_pool2d(out_16, 3, 1, 1, False, False, None)
        out_16 = None
        x_414 = torch.conv2d(
            input_13,
            l_self_modules_features_modules_20_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_13 = l_self_modules_features_modules_20_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_415 = torch.nn.functional.batch_norm(
            x_414,
            l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_414 = l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_20_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_416 = torch.nn.functional.relu(x_415, inplace=True)
        x_415 = None
        out_17 = torch.cat((x_389, x1_2, x2_3, x_416), 1)
        x_389 = x1_2 = x2_3 = x_416 = None
        x_417 = torch.conv2d(
            out_17,
            l_self_modules_features_modules_21_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_21_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_418 = torch.nn.functional.batch_norm(
            x_417,
            l_self_modules_features_modules_21_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_21_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_21_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_21_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_417 = l_self_modules_features_modules_21_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_21_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_21_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_features_modules_21_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_419 = torch.nn.functional.relu(x_418, inplace=True)
        x_418 = None
        x_420 = torch.conv2d(
            out_17,
            l_self_modules_features_modules_21_modules_branch1_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_21_modules_branch1_0_modules_conv_parameters_weight_ = (
            None
        )
        x_421 = torch.nn.functional.batch_norm(
            x_420,
            l_self_modules_features_modules_21_modules_branch1_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_21_modules_branch1_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_21_modules_branch1_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_21_modules_branch1_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_420 = l_self_modules_features_modules_21_modules_branch1_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_21_modules_branch1_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_21_modules_branch1_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_21_modules_branch1_0_modules_bn_parameters_bias_ = (None)
        x_422 = torch.nn.functional.relu(x_421, inplace=True)
        x_421 = None
        x_423 = torch.conv2d(
            x_422,
            l_self_modules_features_modules_21_modules_branch1_1a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_21_modules_branch1_1a_modules_conv_parameters_weight_ = (
            None
        )
        x_424 = torch.nn.functional.batch_norm(
            x_423,
            l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_423 = l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_parameters_weight_ = l_self_modules_features_modules_21_modules_branch1_1a_modules_bn_parameters_bias_ = (None)
        x_425 = torch.nn.functional.relu(x_424, inplace=True)
        x_424 = None
        x_426 = torch.conv2d(
            x_422,
            l_self_modules_features_modules_21_modules_branch1_1b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_422 = l_self_modules_features_modules_21_modules_branch1_1b_modules_conv_parameters_weight_ = (None)
        x_427 = torch.nn.functional.batch_norm(
            x_426,
            l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_426 = l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_parameters_weight_ = l_self_modules_features_modules_21_modules_branch1_1b_modules_bn_parameters_bias_ = (None)
        x_428 = torch.nn.functional.relu(x_427, inplace=True)
        x_427 = None
        x1_3 = torch.cat((x_425, x_428), 1)
        x_425 = x_428 = None
        x_429 = torch.conv2d(
            out_17,
            l_self_modules_features_modules_21_modules_branch2_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_21_modules_branch2_0_modules_conv_parameters_weight_ = (
            None
        )
        x_430 = torch.nn.functional.batch_norm(
            x_429,
            l_self_modules_features_modules_21_modules_branch2_0_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_21_modules_branch2_0_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_21_modules_branch2_0_modules_bn_parameters_weight_,
            l_self_modules_features_modules_21_modules_branch2_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_429 = l_self_modules_features_modules_21_modules_branch2_0_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_21_modules_branch2_0_modules_bn_buffers_running_var_ = l_self_modules_features_modules_21_modules_branch2_0_modules_bn_parameters_weight_ = l_self_modules_features_modules_21_modules_branch2_0_modules_bn_parameters_bias_ = (None)
        x_431 = torch.nn.functional.relu(x_430, inplace=True)
        x_430 = None
        x_432 = torch.conv2d(
            x_431,
            l_self_modules_features_modules_21_modules_branch2_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_431 = l_self_modules_features_modules_21_modules_branch2_1_modules_conv_parameters_weight_ = (None)
        x_433 = torch.nn.functional.batch_norm(
            x_432,
            l_self_modules_features_modules_21_modules_branch2_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_21_modules_branch2_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_21_modules_branch2_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_21_modules_branch2_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_432 = l_self_modules_features_modules_21_modules_branch2_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_21_modules_branch2_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_21_modules_branch2_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_21_modules_branch2_1_modules_bn_parameters_bias_ = (None)
        x_434 = torch.nn.functional.relu(x_433, inplace=True)
        x_433 = None
        x_435 = torch.conv2d(
            x_434,
            l_self_modules_features_modules_21_modules_branch2_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_434 = l_self_modules_features_modules_21_modules_branch2_2_modules_conv_parameters_weight_ = (None)
        x_436 = torch.nn.functional.batch_norm(
            x_435,
            l_self_modules_features_modules_21_modules_branch2_2_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_21_modules_branch2_2_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_21_modules_branch2_2_modules_bn_parameters_weight_,
            l_self_modules_features_modules_21_modules_branch2_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_435 = l_self_modules_features_modules_21_modules_branch2_2_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_21_modules_branch2_2_modules_bn_buffers_running_var_ = l_self_modules_features_modules_21_modules_branch2_2_modules_bn_parameters_weight_ = l_self_modules_features_modules_21_modules_branch2_2_modules_bn_parameters_bias_ = (None)
        x_437 = torch.nn.functional.relu(x_436, inplace=True)
        x_436 = None
        x_438 = torch.conv2d(
            x_437,
            l_self_modules_features_modules_21_modules_branch2_3a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_21_modules_branch2_3a_modules_conv_parameters_weight_ = (
            None
        )
        x_439 = torch.nn.functional.batch_norm(
            x_438,
            l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_parameters_weight_,
            l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_438 = l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_buffers_running_var_ = l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_parameters_weight_ = l_self_modules_features_modules_21_modules_branch2_3a_modules_bn_parameters_bias_ = (None)
        x_440 = torch.nn.functional.relu(x_439, inplace=True)
        x_439 = None
        x_441 = torch.conv2d(
            x_437,
            l_self_modules_features_modules_21_modules_branch2_3b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_437 = l_self_modules_features_modules_21_modules_branch2_3b_modules_conv_parameters_weight_ = (None)
        x_442 = torch.nn.functional.batch_norm(
            x_441,
            l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_parameters_weight_,
            l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_441 = l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_buffers_running_var_ = l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_parameters_weight_ = l_self_modules_features_modules_21_modules_branch2_3b_modules_bn_parameters_bias_ = (None)
        x_443 = torch.nn.functional.relu(x_442, inplace=True)
        x_442 = None
        x2_4 = torch.cat((x_440, x_443), 1)
        x_440 = x_443 = None
        input_14 = torch._C._nn.avg_pool2d(out_17, 3, 1, 1, False, False, None)
        out_17 = None
        x_444 = torch.conv2d(
            input_14,
            l_self_modules_features_modules_21_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_14 = l_self_modules_features_modules_21_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_445 = torch.nn.functional.batch_norm(
            x_444,
            l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_444 = l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_features_modules_21_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_446 = torch.nn.functional.relu(x_445, inplace=True)
        x_445 = None
        out_18 = torch.cat((x_419, x1_3, x2_4, x_446), 1)
        x_419 = x1_3 = x2_4 = x_446 = None
        x_447 = torch.nn.functional.adaptive_avg_pool2d(out_18, 1)
        out_18 = None
        x_448 = x_447.flatten(1, -1)
        x_447 = None
        x_449 = torch.nn.functional.dropout(x_448, 0.0, False, False)
        x_448 = None
        x_450 = torch._C._nn.linear(
            x_449,
            l_self_modules_last_linear_parameters_weight_,
            l_self_modules_last_linear_parameters_bias_,
        )
        x_449 = (
            l_self_modules_last_linear_parameters_weight_
        ) = l_self_modules_last_linear_parameters_bias_ = None
        return (x_450,)
