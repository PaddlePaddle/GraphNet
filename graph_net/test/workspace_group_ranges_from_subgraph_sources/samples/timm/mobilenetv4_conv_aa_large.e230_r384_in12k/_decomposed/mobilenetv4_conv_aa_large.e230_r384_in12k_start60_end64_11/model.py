import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_,
        x_52,
        x_59,
    ):
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_63 = x_62 + x_52
        x_62 = x_52 = None
        return (x_63,)
