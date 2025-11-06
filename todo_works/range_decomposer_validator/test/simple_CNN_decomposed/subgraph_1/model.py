import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        input_3: torch.Tensor,  # Output of subgraph_0
        L_self_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_conv2_parameters_weight_ = (
            L_self_modules_conv2_parameters_weight_
        )
        l_self_modules_conv2_parameters_bias_ = L_self_modules_conv2_parameters_bias_

        # --- Subgraph 1 ---
        # conv2 -> relu -> pool2
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_conv2_parameters_weight_,
            l_self_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = (
            l_self_modules_conv2_parameters_weight_
        ) = l_self_modules_conv2_parameters_bias_ = None
        input_5 = torch.nn.functional.relu(input_4, inplace=True)
        input_4 = None
        input_6 = torch.nn.functional.max_pool2d(input_5, 2, 2, 0, 1, ceil_mode=False)
        input_5 = None

        return (input_6,)
