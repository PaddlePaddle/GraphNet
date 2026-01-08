import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
        x_32,
    ):
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_35 = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        return (x_37,)
