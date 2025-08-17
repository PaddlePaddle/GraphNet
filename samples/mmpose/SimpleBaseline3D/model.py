import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_expand_conv_modules_conv_parameters_weight_ = (
            L_self_modules_backbone_modules_expand_conv_modules_conv_parameters_weight_
        )
        l_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_mean_ = (
            L_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_mean_
        )
        l_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_var_ = (
            L_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_var_
        )
        l_self_modules_backbone_modules_expand_conv_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_expand_conv_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_expand_conv_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_expand_conv_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_bias_
        l_self_modules_head_modules_conv_parameters_weight_ = (
            L_self_modules_head_modules_conv_parameters_weight_
        )
        l_self_modules_head_modules_conv_parameters_bias_ = (
            L_self_modules_head_modules_conv_parameters_bias_
        )
        x = torch.conv1d(
            l_inputs_,
            l_self_modules_backbone_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_inputs_ = (
            l_self_modules_backbone_modules_expand_conv_modules_conv_parameters_weight_
        ) = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_expand_conv_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_expand_conv_modules_bn_parameters_bias_
        ) = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.nn.functional.dropout(x_2, 0.5, False, False)
        x_2 = None
        x_4 = torch.conv1d(
            x_3,
            l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_conv_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_5 = torch.nn.functional.batch_norm(
            x_4,
            l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_4 = l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_bias_ = (None)
        x_6 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        out = torch.nn.functional.dropout(x_6, 0.5, False, False)
        x_6 = None
        x_7 = torch.conv1d(
            out,
            l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_conv_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        out = l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_conv_parameters_weight_ = (None)
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_bias_ = (None)
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        out_1 = torch.nn.functional.dropout(x_9, 0.5, False, False)
        x_9 = None
        res = x_3[(slice(None, None, None), slice(None, None, None), slice(0, 1, None))]
        x_3 = None
        out_2 = out_1 + res
        out_1 = res = None
        x_10 = torch.conv1d(
            out_2,
            l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_conv_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        out_3 = torch.nn.functional.dropout(x_12, 0.5, False, False)
        x_12 = None
        x_13 = torch.conv1d(
            out_3,
            l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_conv_parameters_weight_,
            None,
            (1,),
            (0,),
            (1,),
            1,
        )
        out_3 = l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_conv_parameters_weight_ = (None)
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_bias_ = (None)
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        out_4 = torch.nn.functional.dropout(x_15, 0.5, False, False)
        x_15 = None
        res_1 = out_2[
            (slice(None, None, None), slice(None, None, None), slice(0, 1, None))
        ]
        out_2 = None
        out_5 = out_4 + res_1
        out_4 = res_1 = None
        x_16 = torch.conv1d(
            out_5,
            l_self_modules_head_modules_conv_parameters_weight_,
            l_self_modules_head_modules_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        out_5 = (
            l_self_modules_head_modules_conv_parameters_weight_
        ) = l_self_modules_head_modules_conv_parameters_bias_ = None
        x_17 = x_16.reshape(-1, 16, 3)
        x_16 = None
        return (x_17,)
