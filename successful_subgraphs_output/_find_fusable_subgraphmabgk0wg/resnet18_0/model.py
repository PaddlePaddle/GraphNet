import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        l_l_self_modules_bn1_buffers_running_mean_,
        l_l_self_modules_bn1_buffers_running_var_,
        l_l_self_modules_bn1_parameters_bias_,
        l_l_self_modules_bn1_parameters_weight_,
        x,
    ):
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_l_self_modules_bn1_buffers_running_mean_,
            l_l_self_modules_bn1_buffers_running_var_,
            l_l_self_modules_bn1_parameters_weight_,
            l_l_self_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_l_self_modules_bn1_buffers_running_mean_
        ) = (
            l_l_self_modules_bn1_buffers_running_var_
        ) = (
            l_l_self_modules_bn1_parameters_weight_
        ) = l_l_self_modules_bn1_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        return (x_2,)
