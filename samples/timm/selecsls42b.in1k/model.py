import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv5_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv5_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv5_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv6_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv6_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv6_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv5_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv5_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv5_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv6_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv6_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv6_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv5_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv5_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv5_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv6_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv6_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv6_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv5_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv5_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv5_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv6_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv6_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv6_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv5_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv5_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv5_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv6_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv6_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv6_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv5_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv5_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv5_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv6_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv6_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv6_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_0_parameters_weight_ = (
            L_self_modules_stem_modules_0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_1_buffers_running_mean_ = (
            L_self_modules_stem_modules_1_buffers_running_mean_
        )
        l_self_modules_stem_modules_1_buffers_running_var_ = (
            L_self_modules_stem_modules_1_buffers_running_var_
        )
        l_self_modules_stem_modules_1_parameters_weight_ = (
            L_self_modules_stem_modules_1_parameters_weight_
        )
        l_self_modules_stem_modules_1_parameters_bias_ = (
            L_self_modules_stem_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_0_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_var_ = L_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_var_
        l_self_modules_features_modules_0_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_0_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_var_ = L_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_var_
        l_self_modules_features_modules_0_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_0_modules_conv3_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv3_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_var_ = L_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_var_
        l_self_modules_features_modules_0_modules_conv3_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv3_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv3_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_conv3_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_0_modules_conv4_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv4_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_mean_ = L_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_mean_
        l_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_var_ = L_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_var_
        l_self_modules_features_modules_0_modules_conv4_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv4_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv4_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_conv4_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_0_modules_conv5_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv5_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_mean_ = L_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_mean_
        l_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_var_ = L_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_var_
        l_self_modules_features_modules_0_modules_conv5_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv5_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv5_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_conv5_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_0_modules_conv6_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv6_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_mean_ = L_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_mean_
        l_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_var_ = L_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_var_
        l_self_modules_features_modules_0_modules_conv6_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_conv6_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_conv6_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_conv6_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_conv3_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv3_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_conv3_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv3_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv3_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_conv3_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_conv4_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv4_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_conv4_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv4_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv4_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_conv4_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_conv5_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv5_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_conv5_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv5_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv5_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_conv5_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_conv6_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv6_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_conv6_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv6_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv6_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_conv6_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_conv3_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv3_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv3_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv3_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv3_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_conv3_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_conv4_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv4_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv4_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv4_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv4_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_conv4_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_conv5_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv5_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv5_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv5_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv5_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_conv5_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_conv6_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv6_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv6_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv6_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv6_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_conv6_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_conv3_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv3_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv3_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv3_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv3_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_conv3_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_conv4_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv4_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv4_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv4_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv4_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_conv4_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_conv5_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv5_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv5_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv5_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv5_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_conv5_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_conv6_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv6_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv6_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv6_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv6_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_conv6_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_conv3_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv3_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv3_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv3_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv3_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_conv3_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_conv4_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv4_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv4_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv4_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv4_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_conv4_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_conv5_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv5_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv5_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv5_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv5_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_conv5_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_conv6_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv6_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv6_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv6_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv6_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_conv6_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_conv3_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv3_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv3_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv3_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv3_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_conv3_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_conv4_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv4_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv4_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv4_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv4_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_conv4_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_conv5_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv5_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv5_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv5_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv5_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_conv5_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_conv6_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv6_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv6_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv6_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv6_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_conv6_modules_1_parameters_bias_
        )
        l_self_modules_head_modules_0_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_0_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_0_modules_1_buffers_running_mean_ = (
            L_self_modules_head_modules_0_modules_1_buffers_running_mean_
        )
        l_self_modules_head_modules_0_modules_1_buffers_running_var_ = (
            L_self_modules_head_modules_0_modules_1_buffers_running_var_
        )
        l_self_modules_head_modules_0_modules_1_parameters_weight_ = (
            L_self_modules_head_modules_0_modules_1_parameters_weight_
        )
        l_self_modules_head_modules_0_modules_1_parameters_bias_ = (
            L_self_modules_head_modules_0_modules_1_parameters_bias_
        )
        l_self_modules_head_modules_1_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_1_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_1_modules_1_buffers_running_mean_ = (
            L_self_modules_head_modules_1_modules_1_buffers_running_mean_
        )
        l_self_modules_head_modules_1_modules_1_buffers_running_var_ = (
            L_self_modules_head_modules_1_modules_1_buffers_running_var_
        )
        l_self_modules_head_modules_1_modules_1_parameters_weight_ = (
            L_self_modules_head_modules_1_modules_1_parameters_weight_
        )
        l_self_modules_head_modules_1_modules_1_parameters_bias_ = (
            L_self_modules_head_modules_1_modules_1_parameters_bias_
        )
        l_self_modules_head_modules_2_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_2_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_2_modules_1_buffers_running_mean_ = (
            L_self_modules_head_modules_2_modules_1_buffers_running_mean_
        )
        l_self_modules_head_modules_2_modules_1_buffers_running_var_ = (
            L_self_modules_head_modules_2_modules_1_buffers_running_var_
        )
        l_self_modules_head_modules_2_modules_1_parameters_weight_ = (
            L_self_modules_head_modules_2_modules_1_parameters_weight_
        )
        l_self_modules_head_modules_2_modules_1_parameters_bias_ = (
            L_self_modules_head_modules_2_modules_1_parameters_bias_
        )
        l_self_modules_head_modules_3_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_3_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_3_modules_1_buffers_running_mean_ = (
            L_self_modules_head_modules_3_modules_1_buffers_running_mean_
        )
        l_self_modules_head_modules_3_modules_1_buffers_running_var_ = (
            L_self_modules_head_modules_3_modules_1_buffers_running_var_
        )
        l_self_modules_head_modules_3_modules_1_parameters_weight_ = (
            L_self_modules_head_modules_3_modules_1_parameters_weight_
        )
        l_self_modules_head_modules_3_modules_1_parameters_bias_ = (
            L_self_modules_head_modules_3_modules_1_parameters_bias_
        )
        l_self_modules_fc_parameters_weight_ = L_self_modules_fc_parameters_weight_
        l_self_modules_fc_parameters_bias_ = L_self_modules_fc_parameters_bias_
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_0_parameters_weight_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_stem_modules_1_buffers_running_mean_,
            l_self_modules_stem_modules_1_buffers_running_var_,
            l_self_modules_stem_modules_1_parameters_weight_,
            l_self_modules_stem_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_stem_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_1_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_1_parameters_weight_
        ) = l_self_modules_stem_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_features_modules_0_modules_conv1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = (
            l_self_modules_features_modules_0_modules_conv1_modules_0_parameters_weight_
        ) = None
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_0_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_features_modules_0_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = l_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_0_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_0_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_features_modules_0_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_0_modules_conv2_modules_0_parameters_weight_ = (
            None
        )
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_0_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_features_modules_0_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_0_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_0_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_9 = torch.nn.functional.relu(input_8, inplace=True)
        input_8 = None
        input_10 = torch.conv2d(
            input_9,
            l_self_modules_features_modules_0_modules_conv3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_9 = (
            l_self_modules_features_modules_0_modules_conv3_modules_0_parameters_weight_
        ) = None
        input_11 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_0_modules_conv3_modules_1_parameters_weight_,
            l_self_modules_features_modules_0_modules_conv3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_10 = l_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_0_modules_conv3_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_0_modules_conv3_modules_1_parameters_bias_
        ) = None
        input_12 = torch.nn.functional.relu(input_11, inplace=True)
        input_11 = None
        input_13 = torch.conv2d(
            input_12,
            l_self_modules_features_modules_0_modules_conv4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_0_modules_conv4_modules_0_parameters_weight_ = (
            None
        )
        input_14 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_var_,
            l_self_modules_features_modules_0_modules_conv4_modules_1_parameters_weight_,
            l_self_modules_features_modules_0_modules_conv4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_13 = l_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_mean_ = l_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_0_modules_conv4_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_0_modules_conv4_modules_1_parameters_bias_
        ) = None
        input_15 = torch.nn.functional.relu(input_14, inplace=True)
        input_14 = None
        input_16 = torch.conv2d(
            input_15,
            l_self_modules_features_modules_0_modules_conv5_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_15 = (
            l_self_modules_features_modules_0_modules_conv5_modules_0_parameters_weight_
        ) = None
        input_17 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_var_,
            l_self_modules_features_modules_0_modules_conv5_modules_1_parameters_weight_,
            l_self_modules_features_modules_0_modules_conv5_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_16 = l_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_mean_ = l_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_0_modules_conv5_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_0_modules_conv5_modules_1_parameters_bias_
        ) = None
        input_18 = torch.nn.functional.relu(input_17, inplace=True)
        input_17 = None
        cat = torch.cat([input_6, input_12, input_18], 1)
        input_6 = input_12 = input_18 = None
        input_19 = torch.conv2d(
            cat,
            l_self_modules_features_modules_0_modules_conv6_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = (
            l_self_modules_features_modules_0_modules_conv6_modules_0_parameters_weight_
        ) = None
        input_20 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_var_,
            l_self_modules_features_modules_0_modules_conv6_modules_1_parameters_weight_,
            l_self_modules_features_modules_0_modules_conv6_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_19 = l_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_mean_ = l_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_0_modules_conv6_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_0_modules_conv6_modules_1_parameters_bias_
        ) = None
        input_21 = torch.nn.functional.relu(input_20, inplace=True)
        input_20 = None
        input_22 = torch.conv2d(
            input_21,
            l_self_modules_features_modules_1_modules_conv1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_1_modules_conv1_modules_0_parameters_weight_ = (
            None
        )
        input_23 = torch.nn.functional.batch_norm(
            input_22,
            l_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_22 = l_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_1_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_1_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_24 = torch.nn.functional.relu(input_23, inplace=True)
        input_23 = None
        input_25 = torch.conv2d(
            input_24,
            l_self_modules_features_modules_1_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_1_modules_conv2_modules_0_parameters_weight_ = (
            None
        )
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_25 = l_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_1_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_1_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_27 = torch.nn.functional.relu(input_26, inplace=True)
        input_26 = None
        input_28 = torch.conv2d(
            input_27,
            l_self_modules_features_modules_1_modules_conv3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_27 = (
            l_self_modules_features_modules_1_modules_conv3_modules_0_parameters_weight_
        ) = None
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_conv3_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_conv3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_28 = l_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_1_modules_conv3_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_1_modules_conv3_modules_1_parameters_bias_
        ) = None
        input_30 = torch.nn.functional.relu(input_29, inplace=True)
        input_29 = None
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_features_modules_1_modules_conv4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_1_modules_conv4_modules_0_parameters_weight_ = (
            None
        )
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_conv4_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_conv4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_31 = l_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_1_modules_conv4_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_1_modules_conv4_modules_1_parameters_bias_
        ) = None
        input_33 = torch.nn.functional.relu(input_32, inplace=True)
        input_32 = None
        input_34 = torch.conv2d(
            input_33,
            l_self_modules_features_modules_1_modules_conv5_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_33 = (
            l_self_modules_features_modules_1_modules_conv5_modules_0_parameters_weight_
        ) = None
        input_35 = torch.nn.functional.batch_norm(
            input_34,
            l_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_conv5_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_conv5_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_34 = l_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_1_modules_conv5_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_1_modules_conv5_modules_1_parameters_bias_
        ) = None
        input_36 = torch.nn.functional.relu(input_35, inplace=True)
        input_35 = None
        cat_1 = torch.cat([input_24, input_30, input_36, input_21], 1)
        input_24 = input_30 = input_36 = input_21 = None
        input_37 = torch.conv2d(
            cat_1,
            l_self_modules_features_modules_1_modules_conv6_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = (
            l_self_modules_features_modules_1_modules_conv6_modules_0_parameters_weight_
        ) = None
        input_38 = torch.nn.functional.batch_norm(
            input_37,
            l_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_conv6_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_conv6_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_37 = l_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_1_modules_conv6_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_1_modules_conv6_modules_1_parameters_bias_
        ) = None
        input_39 = torch.nn.functional.relu(input_38, inplace=True)
        input_38 = None
        input_40 = torch.conv2d(
            input_39,
            l_self_modules_features_modules_2_modules_conv1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_39 = (
            l_self_modules_features_modules_2_modules_conv1_modules_0_parameters_weight_
        ) = None
        input_41 = torch.nn.functional.batch_norm(
            input_40,
            l_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_40 = l_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_2_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_2_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_42 = torch.nn.functional.relu(input_41, inplace=True)
        input_41 = None
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_features_modules_2_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_conv2_modules_0_parameters_weight_ = (
            None
        )
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_43 = l_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_2_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_2_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_45 = torch.nn.functional.relu(input_44, inplace=True)
        input_44 = None
        input_46 = torch.conv2d(
            input_45,
            l_self_modules_features_modules_2_modules_conv3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_45 = (
            l_self_modules_features_modules_2_modules_conv3_modules_0_parameters_weight_
        ) = None
        input_47 = torch.nn.functional.batch_norm(
            input_46,
            l_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_46 = l_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_2_modules_conv3_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_2_modules_conv3_modules_1_parameters_bias_
        ) = None
        input_48 = torch.nn.functional.relu(input_47, inplace=True)
        input_47 = None
        input_49 = torch.conv2d(
            input_48,
            l_self_modules_features_modules_2_modules_conv4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_conv4_modules_0_parameters_weight_ = (
            None
        )
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv4_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_49 = l_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_2_modules_conv4_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_2_modules_conv4_modules_1_parameters_bias_
        ) = None
        input_51 = torch.nn.functional.relu(input_50, inplace=True)
        input_50 = None
        input_52 = torch.conv2d(
            input_51,
            l_self_modules_features_modules_2_modules_conv5_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_51 = (
            l_self_modules_features_modules_2_modules_conv5_modules_0_parameters_weight_
        ) = None
        input_53 = torch.nn.functional.batch_norm(
            input_52,
            l_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv5_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv5_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_52 = l_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_2_modules_conv5_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_2_modules_conv5_modules_1_parameters_bias_
        ) = None
        input_54 = torch.nn.functional.relu(input_53, inplace=True)
        input_53 = None
        cat_2 = torch.cat([input_42, input_48, input_54], 1)
        input_42 = input_48 = input_54 = None
        input_55 = torch.conv2d(
            cat_2,
            l_self_modules_features_modules_2_modules_conv6_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = (
            l_self_modules_features_modules_2_modules_conv6_modules_0_parameters_weight_
        ) = None
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv6_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv6_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_55 = l_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_2_modules_conv6_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_2_modules_conv6_modules_1_parameters_bias_
        ) = None
        input_57 = torch.nn.functional.relu(input_56, inplace=True)
        input_56 = None
        input_58 = torch.conv2d(
            input_57,
            l_self_modules_features_modules_3_modules_conv1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_conv1_modules_0_parameters_weight_ = (
            None
        )
        input_59 = torch.nn.functional.batch_norm(
            input_58,
            l_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_58 = l_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_3_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_60 = torch.nn.functional.relu(input_59, inplace=True)
        input_59 = None
        input_61 = torch.conv2d(
            input_60,
            l_self_modules_features_modules_3_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_conv2_modules_0_parameters_weight_ = (
            None
        )
        input_62 = torch.nn.functional.batch_norm(
            input_61,
            l_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_61 = l_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_3_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_63 = torch.nn.functional.relu(input_62, inplace=True)
        input_62 = None
        input_64 = torch.conv2d(
            input_63,
            l_self_modules_features_modules_3_modules_conv3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_63 = (
            l_self_modules_features_modules_3_modules_conv3_modules_0_parameters_weight_
        ) = None
        input_65 = torch.nn.functional.batch_norm(
            input_64,
            l_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_64 = l_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_3_modules_conv3_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_conv3_modules_1_parameters_bias_
        ) = None
        input_66 = torch.nn.functional.relu(input_65, inplace=True)
        input_65 = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_features_modules_3_modules_conv4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_conv4_modules_0_parameters_weight_ = (
            None
        )
        input_68 = torch.nn.functional.batch_norm(
            input_67,
            l_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv4_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_67 = l_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_3_modules_conv4_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_conv4_modules_1_parameters_bias_
        ) = None
        input_69 = torch.nn.functional.relu(input_68, inplace=True)
        input_68 = None
        input_70 = torch.conv2d(
            input_69,
            l_self_modules_features_modules_3_modules_conv5_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_69 = (
            l_self_modules_features_modules_3_modules_conv5_modules_0_parameters_weight_
        ) = None
        input_71 = torch.nn.functional.batch_norm(
            input_70,
            l_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv5_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv5_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_70 = l_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_3_modules_conv5_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_conv5_modules_1_parameters_bias_
        ) = None
        input_72 = torch.nn.functional.relu(input_71, inplace=True)
        input_71 = None
        cat_3 = torch.cat([input_60, input_66, input_72, input_57], 1)
        input_60 = input_66 = input_72 = input_57 = None
        input_73 = torch.conv2d(
            cat_3,
            l_self_modules_features_modules_3_modules_conv6_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = (
            l_self_modules_features_modules_3_modules_conv6_modules_0_parameters_weight_
        ) = None
        input_74 = torch.nn.functional.batch_norm(
            input_73,
            l_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv6_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv6_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_73 = l_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_3_modules_conv6_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_conv6_modules_1_parameters_bias_
        ) = None
        input_75 = torch.nn.functional.relu(input_74, inplace=True)
        input_74 = None
        input_76 = torch.conv2d(
            input_75,
            l_self_modules_features_modules_4_modules_conv1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_75 = (
            l_self_modules_features_modules_4_modules_conv1_modules_0_parameters_weight_
        ) = None
        input_77 = torch.nn.functional.batch_norm(
            input_76,
            l_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_76 = l_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_4_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_4_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_78 = torch.nn.functional.relu(input_77, inplace=True)
        input_77 = None
        input_79 = torch.conv2d(
            input_78,
            l_self_modules_features_modules_4_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_conv2_modules_0_parameters_weight_ = (
            None
        )
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_79 = l_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_4_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_4_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_81 = torch.nn.functional.relu(input_80, inplace=True)
        input_80 = None
        input_82 = torch.conv2d(
            input_81,
            l_self_modules_features_modules_4_modules_conv3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_81 = (
            l_self_modules_features_modules_4_modules_conv3_modules_0_parameters_weight_
        ) = None
        input_83 = torch.nn.functional.batch_norm(
            input_82,
            l_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_82 = l_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_4_modules_conv3_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_4_modules_conv3_modules_1_parameters_bias_
        ) = None
        input_84 = torch.nn.functional.relu(input_83, inplace=True)
        input_83 = None
        input_85 = torch.conv2d(
            input_84,
            l_self_modules_features_modules_4_modules_conv4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_conv4_modules_0_parameters_weight_ = (
            None
        )
        input_86 = torch.nn.functional.batch_norm(
            input_85,
            l_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv4_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_85 = l_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_4_modules_conv4_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_4_modules_conv4_modules_1_parameters_bias_
        ) = None
        input_87 = torch.nn.functional.relu(input_86, inplace=True)
        input_86 = None
        input_88 = torch.conv2d(
            input_87,
            l_self_modules_features_modules_4_modules_conv5_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_87 = (
            l_self_modules_features_modules_4_modules_conv5_modules_0_parameters_weight_
        ) = None
        input_89 = torch.nn.functional.batch_norm(
            input_88,
            l_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv5_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv5_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_88 = l_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_4_modules_conv5_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_4_modules_conv5_modules_1_parameters_bias_
        ) = None
        input_90 = torch.nn.functional.relu(input_89, inplace=True)
        input_89 = None
        cat_4 = torch.cat([input_78, input_84, input_90], 1)
        input_78 = input_84 = input_90 = None
        input_91 = torch.conv2d(
            cat_4,
            l_self_modules_features_modules_4_modules_conv6_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_4 = (
            l_self_modules_features_modules_4_modules_conv6_modules_0_parameters_weight_
        ) = None
        input_92 = torch.nn.functional.batch_norm(
            input_91,
            l_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv6_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv6_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_91 = l_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_4_modules_conv6_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_4_modules_conv6_modules_1_parameters_bias_
        ) = None
        input_93 = torch.nn.functional.relu(input_92, inplace=True)
        input_92 = None
        input_94 = torch.conv2d(
            input_93,
            l_self_modules_features_modules_5_modules_conv1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_conv1_modules_0_parameters_weight_ = (
            None
        )
        input_95 = torch.nn.functional.batch_norm(
            input_94,
            l_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_94 = l_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_5_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_96 = torch.nn.functional.relu(input_95, inplace=True)
        input_95 = None
        input_97 = torch.conv2d(
            input_96,
            l_self_modules_features_modules_5_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_conv2_modules_0_parameters_weight_ = (
            None
        )
        input_98 = torch.nn.functional.batch_norm(
            input_97,
            l_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_97 = l_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_5_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_99 = torch.nn.functional.relu(input_98, inplace=True)
        input_98 = None
        input_100 = torch.conv2d(
            input_99,
            l_self_modules_features_modules_5_modules_conv3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_99 = (
            l_self_modules_features_modules_5_modules_conv3_modules_0_parameters_weight_
        ) = None
        input_101 = torch.nn.functional.batch_norm(
            input_100,
            l_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_100 = l_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_5_modules_conv3_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_conv3_modules_1_parameters_bias_
        ) = None
        input_102 = torch.nn.functional.relu(input_101, inplace=True)
        input_101 = None
        input_103 = torch.conv2d(
            input_102,
            l_self_modules_features_modules_5_modules_conv4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_conv4_modules_0_parameters_weight_ = (
            None
        )
        input_104 = torch.nn.functional.batch_norm(
            input_103,
            l_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv4_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_103 = l_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_5_modules_conv4_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_conv4_modules_1_parameters_bias_
        ) = None
        input_105 = torch.nn.functional.relu(input_104, inplace=True)
        input_104 = None
        input_106 = torch.conv2d(
            input_105,
            l_self_modules_features_modules_5_modules_conv5_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_105 = (
            l_self_modules_features_modules_5_modules_conv5_modules_0_parameters_weight_
        ) = None
        input_107 = torch.nn.functional.batch_norm(
            input_106,
            l_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv5_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv5_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_106 = l_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_5_modules_conv5_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_conv5_modules_1_parameters_bias_
        ) = None
        input_108 = torch.nn.functional.relu(input_107, inplace=True)
        input_107 = None
        cat_5 = torch.cat([input_96, input_102, input_108, input_93], 1)
        input_96 = input_102 = input_108 = input_93 = None
        input_109 = torch.conv2d(
            cat_5,
            l_self_modules_features_modules_5_modules_conv6_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_5 = (
            l_self_modules_features_modules_5_modules_conv6_modules_0_parameters_weight_
        ) = None
        input_110 = torch.nn.functional.batch_norm(
            input_109,
            l_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv6_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv6_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_109 = l_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_var_ = (
            l_self_modules_features_modules_5_modules_conv6_modules_1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_conv6_modules_1_parameters_bias_
        ) = None
        input_111 = torch.nn.functional.relu(input_110, inplace=True)
        input_110 = None
        input_112 = torch.conv2d(
            input_111,
            l_self_modules_head_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_111 = l_self_modules_head_modules_0_modules_0_parameters_weight_ = None
        input_113 = torch.nn.functional.batch_norm(
            input_112,
            l_self_modules_head_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_0_modules_1_buffers_running_var_,
            l_self_modules_head_modules_0_modules_1_parameters_weight_,
            l_self_modules_head_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_112 = (
            l_self_modules_head_modules_0_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_0_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_0_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_0_modules_1_parameters_bias_ = None
        input_114 = torch.nn.functional.relu(input_113, inplace=True)
        input_113 = None
        input_115 = torch.conv2d(
            input_114,
            l_self_modules_head_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_114 = l_self_modules_head_modules_1_modules_0_parameters_weight_ = None
        input_116 = torch.nn.functional.batch_norm(
            input_115,
            l_self_modules_head_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_1_modules_1_buffers_running_var_,
            l_self_modules_head_modules_1_modules_1_parameters_weight_,
            l_self_modules_head_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_115 = (
            l_self_modules_head_modules_1_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_1_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_1_modules_1_parameters_bias_ = None
        input_117 = torch.nn.functional.relu(input_116, inplace=True)
        input_116 = None
        input_118 = torch.conv2d(
            input_117,
            l_self_modules_head_modules_2_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_117 = l_self_modules_head_modules_2_modules_0_parameters_weight_ = None
        input_119 = torch.nn.functional.batch_norm(
            input_118,
            l_self_modules_head_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_2_modules_1_buffers_running_var_,
            l_self_modules_head_modules_2_modules_1_parameters_weight_,
            l_self_modules_head_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_118 = (
            l_self_modules_head_modules_2_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_2_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_2_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_2_modules_1_parameters_bias_ = None
        input_120 = torch.nn.functional.relu(input_119, inplace=True)
        input_119 = None
        input_121 = torch.conv2d(
            input_120,
            l_self_modules_head_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_120 = l_self_modules_head_modules_3_modules_0_parameters_weight_ = None
        input_122 = torch.nn.functional.batch_norm(
            input_121,
            l_self_modules_head_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_3_modules_1_buffers_running_var_,
            l_self_modules_head_modules_3_modules_1_parameters_weight_,
            l_self_modules_head_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_121 = (
            l_self_modules_head_modules_3_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_3_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_3_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_3_modules_1_parameters_bias_ = None
        input_123 = torch.nn.functional.relu(input_122, inplace=True)
        input_122 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_123, 1)
        input_123 = None
        x_1 = x.flatten(1, -1)
        x = None
        x_2 = torch.nn.functional.dropout(x_1, 0.0, False, False)
        x_1 = None
        x_3 = torch._C._nn.linear(
            x_2,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_2 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_3,)
