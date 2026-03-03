import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_,
        x_54,
    ):
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_57 = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        return (x_59,)
