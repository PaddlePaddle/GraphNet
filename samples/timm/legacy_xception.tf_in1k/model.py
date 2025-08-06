import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block1_modules_rep_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block1_modules_rep_modules_0_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block1_modules_rep_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_block1_modules_rep_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_block1_modules_rep_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block1_modules_rep_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block1_modules_rep_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block1_modules_rep_modules_3_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block1_modules_rep_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_block1_modules_rep_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_block1_modules_rep_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block1_modules_rep_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block1_modules_skip_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block1_modules_skipbn_buffers_running_mean_: torch.Tensor,
        L_self_modules_block1_modules_skipbn_buffers_running_var_: torch.Tensor,
        L_self_modules_block1_modules_skipbn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block1_modules_skipbn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block2_modules_rep_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block2_modules_rep_modules_1_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block2_modules_rep_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_block2_modules_rep_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_block2_modules_rep_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block2_modules_rep_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block2_modules_rep_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block2_modules_rep_modules_4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block2_modules_rep_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_block2_modules_rep_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_block2_modules_rep_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block2_modules_rep_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block2_modules_skip_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block2_modules_skipbn_buffers_running_mean_: torch.Tensor,
        L_self_modules_block2_modules_skipbn_buffers_running_var_: torch.Tensor,
        L_self_modules_block2_modules_skipbn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block2_modules_skipbn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block3_modules_rep_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block3_modules_rep_modules_1_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block3_modules_rep_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_block3_modules_rep_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_block3_modules_rep_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block3_modules_rep_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block3_modules_rep_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block3_modules_rep_modules_4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block3_modules_rep_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_block3_modules_rep_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_block3_modules_rep_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block3_modules_rep_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block3_modules_skip_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block3_modules_skipbn_buffers_running_mean_: torch.Tensor,
        L_self_modules_block3_modules_skipbn_buffers_running_var_: torch.Tensor,
        L_self_modules_block3_modules_skipbn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block3_modules_skipbn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_1_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_block4_modules_rep_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_block4_modules_rep_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_block4_modules_rep_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_block4_modules_rep_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_7_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_block4_modules_rep_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_block4_modules_rep_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block4_modules_rep_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_1_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_block5_modules_rep_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_block5_modules_rep_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_block5_modules_rep_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_block5_modules_rep_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_7_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_block5_modules_rep_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_block5_modules_rep_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block5_modules_rep_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_1_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_block6_modules_rep_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_block6_modules_rep_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_block6_modules_rep_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_block6_modules_rep_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_7_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_block6_modules_rep_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_block6_modules_rep_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block6_modules_rep_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_1_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_block7_modules_rep_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_block7_modules_rep_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_block7_modules_rep_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_block7_modules_rep_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_7_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_block7_modules_rep_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_block7_modules_rep_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block7_modules_rep_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_1_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_block8_modules_rep_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_block8_modules_rep_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_block8_modules_rep_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_block8_modules_rep_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_7_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_block8_modules_rep_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_block8_modules_rep_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_rep_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_1_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_block9_modules_rep_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_block9_modules_rep_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_block9_modules_rep_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_block9_modules_rep_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_7_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_block9_modules_rep_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_block9_modules_rep_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block9_modules_rep_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_1_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_block10_modules_rep_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_block10_modules_rep_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_block10_modules_rep_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_block10_modules_rep_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_7_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_block10_modules_rep_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_block10_modules_rep_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block10_modules_rep_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_1_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_block11_modules_rep_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_block11_modules_rep_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_block11_modules_rep_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_block11_modules_rep_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_7_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_block11_modules_rep_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_block11_modules_rep_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block11_modules_rep_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block12_modules_rep_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block12_modules_rep_modules_1_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block12_modules_rep_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_block12_modules_rep_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_block12_modules_rep_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block12_modules_rep_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block12_modules_rep_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block12_modules_rep_modules_4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block12_modules_rep_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_block12_modules_rep_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_block12_modules_rep_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block12_modules_rep_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block12_modules_skip_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block12_modules_skipbn_buffers_running_mean_: torch.Tensor,
        L_self_modules_block12_modules_skipbn_buffers_running_var_: torch.Tensor,
        L_self_modules_block12_modules_skipbn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block12_modules_skipbn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv3_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv4_modules_pointwise_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn4_buffers_running_mean_: torch.Tensor,
        L_self_modules_bn4_buffers_running_var_: torch.Tensor,
        L_self_modules_bn4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_conv1_parameters_weight_ = (
            L_self_modules_conv1_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_bn1_buffers_running_mean_ = (
            L_self_modules_bn1_buffers_running_mean_
        )
        l_self_modules_bn1_buffers_running_var_ = (
            L_self_modules_bn1_buffers_running_var_
        )
        l_self_modules_bn1_parameters_weight_ = L_self_modules_bn1_parameters_weight_
        l_self_modules_bn1_parameters_bias_ = L_self_modules_bn1_parameters_bias_
        l_self_modules_conv2_parameters_weight_ = (
            L_self_modules_conv2_parameters_weight_
        )
        l_self_modules_bn2_buffers_running_mean_ = (
            L_self_modules_bn2_buffers_running_mean_
        )
        l_self_modules_bn2_buffers_running_var_ = (
            L_self_modules_bn2_buffers_running_var_
        )
        l_self_modules_bn2_parameters_weight_ = L_self_modules_bn2_parameters_weight_
        l_self_modules_bn2_parameters_bias_ = L_self_modules_bn2_parameters_bias_
        l_self_modules_block1_modules_rep_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_block1_modules_rep_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_block1_modules_rep_modules_0_modules_pointwise_parameters_weight_ = L_self_modules_block1_modules_rep_modules_0_modules_pointwise_parameters_weight_
        l_self_modules_block1_modules_rep_modules_1_buffers_running_mean_ = (
            L_self_modules_block1_modules_rep_modules_1_buffers_running_mean_
        )
        l_self_modules_block1_modules_rep_modules_1_buffers_running_var_ = (
            L_self_modules_block1_modules_rep_modules_1_buffers_running_var_
        )
        l_self_modules_block1_modules_rep_modules_1_parameters_weight_ = (
            L_self_modules_block1_modules_rep_modules_1_parameters_weight_
        )
        l_self_modules_block1_modules_rep_modules_1_parameters_bias_ = (
            L_self_modules_block1_modules_rep_modules_1_parameters_bias_
        )
        l_self_modules_block1_modules_rep_modules_3_modules_conv1_parameters_weight_ = (
            L_self_modules_block1_modules_rep_modules_3_modules_conv1_parameters_weight_
        )
        l_self_modules_block1_modules_rep_modules_3_modules_pointwise_parameters_weight_ = L_self_modules_block1_modules_rep_modules_3_modules_pointwise_parameters_weight_
        l_self_modules_block1_modules_rep_modules_4_buffers_running_mean_ = (
            L_self_modules_block1_modules_rep_modules_4_buffers_running_mean_
        )
        l_self_modules_block1_modules_rep_modules_4_buffers_running_var_ = (
            L_self_modules_block1_modules_rep_modules_4_buffers_running_var_
        )
        l_self_modules_block1_modules_rep_modules_4_parameters_weight_ = (
            L_self_modules_block1_modules_rep_modules_4_parameters_weight_
        )
        l_self_modules_block1_modules_rep_modules_4_parameters_bias_ = (
            L_self_modules_block1_modules_rep_modules_4_parameters_bias_
        )
        l_self_modules_block1_modules_skip_parameters_weight_ = (
            L_self_modules_block1_modules_skip_parameters_weight_
        )
        l_self_modules_block1_modules_skipbn_buffers_running_mean_ = (
            L_self_modules_block1_modules_skipbn_buffers_running_mean_
        )
        l_self_modules_block1_modules_skipbn_buffers_running_var_ = (
            L_self_modules_block1_modules_skipbn_buffers_running_var_
        )
        l_self_modules_block1_modules_skipbn_parameters_weight_ = (
            L_self_modules_block1_modules_skipbn_parameters_weight_
        )
        l_self_modules_block1_modules_skipbn_parameters_bias_ = (
            L_self_modules_block1_modules_skipbn_parameters_bias_
        )
        l_self_modules_block2_modules_rep_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_block2_modules_rep_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_block2_modules_rep_modules_1_modules_pointwise_parameters_weight_ = L_self_modules_block2_modules_rep_modules_1_modules_pointwise_parameters_weight_
        l_self_modules_block2_modules_rep_modules_2_buffers_running_mean_ = (
            L_self_modules_block2_modules_rep_modules_2_buffers_running_mean_
        )
        l_self_modules_block2_modules_rep_modules_2_buffers_running_var_ = (
            L_self_modules_block2_modules_rep_modules_2_buffers_running_var_
        )
        l_self_modules_block2_modules_rep_modules_2_parameters_weight_ = (
            L_self_modules_block2_modules_rep_modules_2_parameters_weight_
        )
        l_self_modules_block2_modules_rep_modules_2_parameters_bias_ = (
            L_self_modules_block2_modules_rep_modules_2_parameters_bias_
        )
        l_self_modules_block2_modules_rep_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_block2_modules_rep_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_block2_modules_rep_modules_4_modules_pointwise_parameters_weight_ = L_self_modules_block2_modules_rep_modules_4_modules_pointwise_parameters_weight_
        l_self_modules_block2_modules_rep_modules_5_buffers_running_mean_ = (
            L_self_modules_block2_modules_rep_modules_5_buffers_running_mean_
        )
        l_self_modules_block2_modules_rep_modules_5_buffers_running_var_ = (
            L_self_modules_block2_modules_rep_modules_5_buffers_running_var_
        )
        l_self_modules_block2_modules_rep_modules_5_parameters_weight_ = (
            L_self_modules_block2_modules_rep_modules_5_parameters_weight_
        )
        l_self_modules_block2_modules_rep_modules_5_parameters_bias_ = (
            L_self_modules_block2_modules_rep_modules_5_parameters_bias_
        )
        l_self_modules_block2_modules_skip_parameters_weight_ = (
            L_self_modules_block2_modules_skip_parameters_weight_
        )
        l_self_modules_block2_modules_skipbn_buffers_running_mean_ = (
            L_self_modules_block2_modules_skipbn_buffers_running_mean_
        )
        l_self_modules_block2_modules_skipbn_buffers_running_var_ = (
            L_self_modules_block2_modules_skipbn_buffers_running_var_
        )
        l_self_modules_block2_modules_skipbn_parameters_weight_ = (
            L_self_modules_block2_modules_skipbn_parameters_weight_
        )
        l_self_modules_block2_modules_skipbn_parameters_bias_ = (
            L_self_modules_block2_modules_skipbn_parameters_bias_
        )
        l_self_modules_block3_modules_rep_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_block3_modules_rep_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_block3_modules_rep_modules_1_modules_pointwise_parameters_weight_ = L_self_modules_block3_modules_rep_modules_1_modules_pointwise_parameters_weight_
        l_self_modules_block3_modules_rep_modules_2_buffers_running_mean_ = (
            L_self_modules_block3_modules_rep_modules_2_buffers_running_mean_
        )
        l_self_modules_block3_modules_rep_modules_2_buffers_running_var_ = (
            L_self_modules_block3_modules_rep_modules_2_buffers_running_var_
        )
        l_self_modules_block3_modules_rep_modules_2_parameters_weight_ = (
            L_self_modules_block3_modules_rep_modules_2_parameters_weight_
        )
        l_self_modules_block3_modules_rep_modules_2_parameters_bias_ = (
            L_self_modules_block3_modules_rep_modules_2_parameters_bias_
        )
        l_self_modules_block3_modules_rep_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_block3_modules_rep_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_block3_modules_rep_modules_4_modules_pointwise_parameters_weight_ = L_self_modules_block3_modules_rep_modules_4_modules_pointwise_parameters_weight_
        l_self_modules_block3_modules_rep_modules_5_buffers_running_mean_ = (
            L_self_modules_block3_modules_rep_modules_5_buffers_running_mean_
        )
        l_self_modules_block3_modules_rep_modules_5_buffers_running_var_ = (
            L_self_modules_block3_modules_rep_modules_5_buffers_running_var_
        )
        l_self_modules_block3_modules_rep_modules_5_parameters_weight_ = (
            L_self_modules_block3_modules_rep_modules_5_parameters_weight_
        )
        l_self_modules_block3_modules_rep_modules_5_parameters_bias_ = (
            L_self_modules_block3_modules_rep_modules_5_parameters_bias_
        )
        l_self_modules_block3_modules_skip_parameters_weight_ = (
            L_self_modules_block3_modules_skip_parameters_weight_
        )
        l_self_modules_block3_modules_skipbn_buffers_running_mean_ = (
            L_self_modules_block3_modules_skipbn_buffers_running_mean_
        )
        l_self_modules_block3_modules_skipbn_buffers_running_var_ = (
            L_self_modules_block3_modules_skipbn_buffers_running_var_
        )
        l_self_modules_block3_modules_skipbn_parameters_weight_ = (
            L_self_modules_block3_modules_skipbn_parameters_weight_
        )
        l_self_modules_block3_modules_skipbn_parameters_bias_ = (
            L_self_modules_block3_modules_skipbn_parameters_bias_
        )
        l_self_modules_block4_modules_rep_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_block4_modules_rep_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_block4_modules_rep_modules_1_modules_pointwise_parameters_weight_ = L_self_modules_block4_modules_rep_modules_1_modules_pointwise_parameters_weight_
        l_self_modules_block4_modules_rep_modules_2_buffers_running_mean_ = (
            L_self_modules_block4_modules_rep_modules_2_buffers_running_mean_
        )
        l_self_modules_block4_modules_rep_modules_2_buffers_running_var_ = (
            L_self_modules_block4_modules_rep_modules_2_buffers_running_var_
        )
        l_self_modules_block4_modules_rep_modules_2_parameters_weight_ = (
            L_self_modules_block4_modules_rep_modules_2_parameters_weight_
        )
        l_self_modules_block4_modules_rep_modules_2_parameters_bias_ = (
            L_self_modules_block4_modules_rep_modules_2_parameters_bias_
        )
        l_self_modules_block4_modules_rep_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_block4_modules_rep_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_block4_modules_rep_modules_4_modules_pointwise_parameters_weight_ = L_self_modules_block4_modules_rep_modules_4_modules_pointwise_parameters_weight_
        l_self_modules_block4_modules_rep_modules_5_buffers_running_mean_ = (
            L_self_modules_block4_modules_rep_modules_5_buffers_running_mean_
        )
        l_self_modules_block4_modules_rep_modules_5_buffers_running_var_ = (
            L_self_modules_block4_modules_rep_modules_5_buffers_running_var_
        )
        l_self_modules_block4_modules_rep_modules_5_parameters_weight_ = (
            L_self_modules_block4_modules_rep_modules_5_parameters_weight_
        )
        l_self_modules_block4_modules_rep_modules_5_parameters_bias_ = (
            L_self_modules_block4_modules_rep_modules_5_parameters_bias_
        )
        l_self_modules_block4_modules_rep_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_block4_modules_rep_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_block4_modules_rep_modules_7_modules_pointwise_parameters_weight_ = L_self_modules_block4_modules_rep_modules_7_modules_pointwise_parameters_weight_
        l_self_modules_block4_modules_rep_modules_8_buffers_running_mean_ = (
            L_self_modules_block4_modules_rep_modules_8_buffers_running_mean_
        )
        l_self_modules_block4_modules_rep_modules_8_buffers_running_var_ = (
            L_self_modules_block4_modules_rep_modules_8_buffers_running_var_
        )
        l_self_modules_block4_modules_rep_modules_8_parameters_weight_ = (
            L_self_modules_block4_modules_rep_modules_8_parameters_weight_
        )
        l_self_modules_block4_modules_rep_modules_8_parameters_bias_ = (
            L_self_modules_block4_modules_rep_modules_8_parameters_bias_
        )
        l_self_modules_block5_modules_rep_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_block5_modules_rep_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_block5_modules_rep_modules_1_modules_pointwise_parameters_weight_ = L_self_modules_block5_modules_rep_modules_1_modules_pointwise_parameters_weight_
        l_self_modules_block5_modules_rep_modules_2_buffers_running_mean_ = (
            L_self_modules_block5_modules_rep_modules_2_buffers_running_mean_
        )
        l_self_modules_block5_modules_rep_modules_2_buffers_running_var_ = (
            L_self_modules_block5_modules_rep_modules_2_buffers_running_var_
        )
        l_self_modules_block5_modules_rep_modules_2_parameters_weight_ = (
            L_self_modules_block5_modules_rep_modules_2_parameters_weight_
        )
        l_self_modules_block5_modules_rep_modules_2_parameters_bias_ = (
            L_self_modules_block5_modules_rep_modules_2_parameters_bias_
        )
        l_self_modules_block5_modules_rep_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_block5_modules_rep_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_block5_modules_rep_modules_4_modules_pointwise_parameters_weight_ = L_self_modules_block5_modules_rep_modules_4_modules_pointwise_parameters_weight_
        l_self_modules_block5_modules_rep_modules_5_buffers_running_mean_ = (
            L_self_modules_block5_modules_rep_modules_5_buffers_running_mean_
        )
        l_self_modules_block5_modules_rep_modules_5_buffers_running_var_ = (
            L_self_modules_block5_modules_rep_modules_5_buffers_running_var_
        )
        l_self_modules_block5_modules_rep_modules_5_parameters_weight_ = (
            L_self_modules_block5_modules_rep_modules_5_parameters_weight_
        )
        l_self_modules_block5_modules_rep_modules_5_parameters_bias_ = (
            L_self_modules_block5_modules_rep_modules_5_parameters_bias_
        )
        l_self_modules_block5_modules_rep_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_block5_modules_rep_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_block5_modules_rep_modules_7_modules_pointwise_parameters_weight_ = L_self_modules_block5_modules_rep_modules_7_modules_pointwise_parameters_weight_
        l_self_modules_block5_modules_rep_modules_8_buffers_running_mean_ = (
            L_self_modules_block5_modules_rep_modules_8_buffers_running_mean_
        )
        l_self_modules_block5_modules_rep_modules_8_buffers_running_var_ = (
            L_self_modules_block5_modules_rep_modules_8_buffers_running_var_
        )
        l_self_modules_block5_modules_rep_modules_8_parameters_weight_ = (
            L_self_modules_block5_modules_rep_modules_8_parameters_weight_
        )
        l_self_modules_block5_modules_rep_modules_8_parameters_bias_ = (
            L_self_modules_block5_modules_rep_modules_8_parameters_bias_
        )
        l_self_modules_block6_modules_rep_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_block6_modules_rep_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_block6_modules_rep_modules_1_modules_pointwise_parameters_weight_ = L_self_modules_block6_modules_rep_modules_1_modules_pointwise_parameters_weight_
        l_self_modules_block6_modules_rep_modules_2_buffers_running_mean_ = (
            L_self_modules_block6_modules_rep_modules_2_buffers_running_mean_
        )
        l_self_modules_block6_modules_rep_modules_2_buffers_running_var_ = (
            L_self_modules_block6_modules_rep_modules_2_buffers_running_var_
        )
        l_self_modules_block6_modules_rep_modules_2_parameters_weight_ = (
            L_self_modules_block6_modules_rep_modules_2_parameters_weight_
        )
        l_self_modules_block6_modules_rep_modules_2_parameters_bias_ = (
            L_self_modules_block6_modules_rep_modules_2_parameters_bias_
        )
        l_self_modules_block6_modules_rep_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_block6_modules_rep_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_block6_modules_rep_modules_4_modules_pointwise_parameters_weight_ = L_self_modules_block6_modules_rep_modules_4_modules_pointwise_parameters_weight_
        l_self_modules_block6_modules_rep_modules_5_buffers_running_mean_ = (
            L_self_modules_block6_modules_rep_modules_5_buffers_running_mean_
        )
        l_self_modules_block6_modules_rep_modules_5_buffers_running_var_ = (
            L_self_modules_block6_modules_rep_modules_5_buffers_running_var_
        )
        l_self_modules_block6_modules_rep_modules_5_parameters_weight_ = (
            L_self_modules_block6_modules_rep_modules_5_parameters_weight_
        )
        l_self_modules_block6_modules_rep_modules_5_parameters_bias_ = (
            L_self_modules_block6_modules_rep_modules_5_parameters_bias_
        )
        l_self_modules_block6_modules_rep_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_block6_modules_rep_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_block6_modules_rep_modules_7_modules_pointwise_parameters_weight_ = L_self_modules_block6_modules_rep_modules_7_modules_pointwise_parameters_weight_
        l_self_modules_block6_modules_rep_modules_8_buffers_running_mean_ = (
            L_self_modules_block6_modules_rep_modules_8_buffers_running_mean_
        )
        l_self_modules_block6_modules_rep_modules_8_buffers_running_var_ = (
            L_self_modules_block6_modules_rep_modules_8_buffers_running_var_
        )
        l_self_modules_block6_modules_rep_modules_8_parameters_weight_ = (
            L_self_modules_block6_modules_rep_modules_8_parameters_weight_
        )
        l_self_modules_block6_modules_rep_modules_8_parameters_bias_ = (
            L_self_modules_block6_modules_rep_modules_8_parameters_bias_
        )
        l_self_modules_block7_modules_rep_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_block7_modules_rep_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_block7_modules_rep_modules_1_modules_pointwise_parameters_weight_ = L_self_modules_block7_modules_rep_modules_1_modules_pointwise_parameters_weight_
        l_self_modules_block7_modules_rep_modules_2_buffers_running_mean_ = (
            L_self_modules_block7_modules_rep_modules_2_buffers_running_mean_
        )
        l_self_modules_block7_modules_rep_modules_2_buffers_running_var_ = (
            L_self_modules_block7_modules_rep_modules_2_buffers_running_var_
        )
        l_self_modules_block7_modules_rep_modules_2_parameters_weight_ = (
            L_self_modules_block7_modules_rep_modules_2_parameters_weight_
        )
        l_self_modules_block7_modules_rep_modules_2_parameters_bias_ = (
            L_self_modules_block7_modules_rep_modules_2_parameters_bias_
        )
        l_self_modules_block7_modules_rep_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_block7_modules_rep_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_block7_modules_rep_modules_4_modules_pointwise_parameters_weight_ = L_self_modules_block7_modules_rep_modules_4_modules_pointwise_parameters_weight_
        l_self_modules_block7_modules_rep_modules_5_buffers_running_mean_ = (
            L_self_modules_block7_modules_rep_modules_5_buffers_running_mean_
        )
        l_self_modules_block7_modules_rep_modules_5_buffers_running_var_ = (
            L_self_modules_block7_modules_rep_modules_5_buffers_running_var_
        )
        l_self_modules_block7_modules_rep_modules_5_parameters_weight_ = (
            L_self_modules_block7_modules_rep_modules_5_parameters_weight_
        )
        l_self_modules_block7_modules_rep_modules_5_parameters_bias_ = (
            L_self_modules_block7_modules_rep_modules_5_parameters_bias_
        )
        l_self_modules_block7_modules_rep_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_block7_modules_rep_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_block7_modules_rep_modules_7_modules_pointwise_parameters_weight_ = L_self_modules_block7_modules_rep_modules_7_modules_pointwise_parameters_weight_
        l_self_modules_block7_modules_rep_modules_8_buffers_running_mean_ = (
            L_self_modules_block7_modules_rep_modules_8_buffers_running_mean_
        )
        l_self_modules_block7_modules_rep_modules_8_buffers_running_var_ = (
            L_self_modules_block7_modules_rep_modules_8_buffers_running_var_
        )
        l_self_modules_block7_modules_rep_modules_8_parameters_weight_ = (
            L_self_modules_block7_modules_rep_modules_8_parameters_weight_
        )
        l_self_modules_block7_modules_rep_modules_8_parameters_bias_ = (
            L_self_modules_block7_modules_rep_modules_8_parameters_bias_
        )
        l_self_modules_block8_modules_rep_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_block8_modules_rep_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_block8_modules_rep_modules_1_modules_pointwise_parameters_weight_ = L_self_modules_block8_modules_rep_modules_1_modules_pointwise_parameters_weight_
        l_self_modules_block8_modules_rep_modules_2_buffers_running_mean_ = (
            L_self_modules_block8_modules_rep_modules_2_buffers_running_mean_
        )
        l_self_modules_block8_modules_rep_modules_2_buffers_running_var_ = (
            L_self_modules_block8_modules_rep_modules_2_buffers_running_var_
        )
        l_self_modules_block8_modules_rep_modules_2_parameters_weight_ = (
            L_self_modules_block8_modules_rep_modules_2_parameters_weight_
        )
        l_self_modules_block8_modules_rep_modules_2_parameters_bias_ = (
            L_self_modules_block8_modules_rep_modules_2_parameters_bias_
        )
        l_self_modules_block8_modules_rep_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_block8_modules_rep_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_block8_modules_rep_modules_4_modules_pointwise_parameters_weight_ = L_self_modules_block8_modules_rep_modules_4_modules_pointwise_parameters_weight_
        l_self_modules_block8_modules_rep_modules_5_buffers_running_mean_ = (
            L_self_modules_block8_modules_rep_modules_5_buffers_running_mean_
        )
        l_self_modules_block8_modules_rep_modules_5_buffers_running_var_ = (
            L_self_modules_block8_modules_rep_modules_5_buffers_running_var_
        )
        l_self_modules_block8_modules_rep_modules_5_parameters_weight_ = (
            L_self_modules_block8_modules_rep_modules_5_parameters_weight_
        )
        l_self_modules_block8_modules_rep_modules_5_parameters_bias_ = (
            L_self_modules_block8_modules_rep_modules_5_parameters_bias_
        )
        l_self_modules_block8_modules_rep_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_block8_modules_rep_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_block8_modules_rep_modules_7_modules_pointwise_parameters_weight_ = L_self_modules_block8_modules_rep_modules_7_modules_pointwise_parameters_weight_
        l_self_modules_block8_modules_rep_modules_8_buffers_running_mean_ = (
            L_self_modules_block8_modules_rep_modules_8_buffers_running_mean_
        )
        l_self_modules_block8_modules_rep_modules_8_buffers_running_var_ = (
            L_self_modules_block8_modules_rep_modules_8_buffers_running_var_
        )
        l_self_modules_block8_modules_rep_modules_8_parameters_weight_ = (
            L_self_modules_block8_modules_rep_modules_8_parameters_weight_
        )
        l_self_modules_block8_modules_rep_modules_8_parameters_bias_ = (
            L_self_modules_block8_modules_rep_modules_8_parameters_bias_
        )
        l_self_modules_block9_modules_rep_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_block9_modules_rep_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_block9_modules_rep_modules_1_modules_pointwise_parameters_weight_ = L_self_modules_block9_modules_rep_modules_1_modules_pointwise_parameters_weight_
        l_self_modules_block9_modules_rep_modules_2_buffers_running_mean_ = (
            L_self_modules_block9_modules_rep_modules_2_buffers_running_mean_
        )
        l_self_modules_block9_modules_rep_modules_2_buffers_running_var_ = (
            L_self_modules_block9_modules_rep_modules_2_buffers_running_var_
        )
        l_self_modules_block9_modules_rep_modules_2_parameters_weight_ = (
            L_self_modules_block9_modules_rep_modules_2_parameters_weight_
        )
        l_self_modules_block9_modules_rep_modules_2_parameters_bias_ = (
            L_self_modules_block9_modules_rep_modules_2_parameters_bias_
        )
        l_self_modules_block9_modules_rep_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_block9_modules_rep_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_block9_modules_rep_modules_4_modules_pointwise_parameters_weight_ = L_self_modules_block9_modules_rep_modules_4_modules_pointwise_parameters_weight_
        l_self_modules_block9_modules_rep_modules_5_buffers_running_mean_ = (
            L_self_modules_block9_modules_rep_modules_5_buffers_running_mean_
        )
        l_self_modules_block9_modules_rep_modules_5_buffers_running_var_ = (
            L_self_modules_block9_modules_rep_modules_5_buffers_running_var_
        )
        l_self_modules_block9_modules_rep_modules_5_parameters_weight_ = (
            L_self_modules_block9_modules_rep_modules_5_parameters_weight_
        )
        l_self_modules_block9_modules_rep_modules_5_parameters_bias_ = (
            L_self_modules_block9_modules_rep_modules_5_parameters_bias_
        )
        l_self_modules_block9_modules_rep_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_block9_modules_rep_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_block9_modules_rep_modules_7_modules_pointwise_parameters_weight_ = L_self_modules_block9_modules_rep_modules_7_modules_pointwise_parameters_weight_
        l_self_modules_block9_modules_rep_modules_8_buffers_running_mean_ = (
            L_self_modules_block9_modules_rep_modules_8_buffers_running_mean_
        )
        l_self_modules_block9_modules_rep_modules_8_buffers_running_var_ = (
            L_self_modules_block9_modules_rep_modules_8_buffers_running_var_
        )
        l_self_modules_block9_modules_rep_modules_8_parameters_weight_ = (
            L_self_modules_block9_modules_rep_modules_8_parameters_weight_
        )
        l_self_modules_block9_modules_rep_modules_8_parameters_bias_ = (
            L_self_modules_block9_modules_rep_modules_8_parameters_bias_
        )
        l_self_modules_block10_modules_rep_modules_1_modules_conv1_parameters_weight_ = L_self_modules_block10_modules_rep_modules_1_modules_conv1_parameters_weight_
        l_self_modules_block10_modules_rep_modules_1_modules_pointwise_parameters_weight_ = L_self_modules_block10_modules_rep_modules_1_modules_pointwise_parameters_weight_
        l_self_modules_block10_modules_rep_modules_2_buffers_running_mean_ = (
            L_self_modules_block10_modules_rep_modules_2_buffers_running_mean_
        )
        l_self_modules_block10_modules_rep_modules_2_buffers_running_var_ = (
            L_self_modules_block10_modules_rep_modules_2_buffers_running_var_
        )
        l_self_modules_block10_modules_rep_modules_2_parameters_weight_ = (
            L_self_modules_block10_modules_rep_modules_2_parameters_weight_
        )
        l_self_modules_block10_modules_rep_modules_2_parameters_bias_ = (
            L_self_modules_block10_modules_rep_modules_2_parameters_bias_
        )
        l_self_modules_block10_modules_rep_modules_4_modules_conv1_parameters_weight_ = L_self_modules_block10_modules_rep_modules_4_modules_conv1_parameters_weight_
        l_self_modules_block10_modules_rep_modules_4_modules_pointwise_parameters_weight_ = L_self_modules_block10_modules_rep_modules_4_modules_pointwise_parameters_weight_
        l_self_modules_block10_modules_rep_modules_5_buffers_running_mean_ = (
            L_self_modules_block10_modules_rep_modules_5_buffers_running_mean_
        )
        l_self_modules_block10_modules_rep_modules_5_buffers_running_var_ = (
            L_self_modules_block10_modules_rep_modules_5_buffers_running_var_
        )
        l_self_modules_block10_modules_rep_modules_5_parameters_weight_ = (
            L_self_modules_block10_modules_rep_modules_5_parameters_weight_
        )
        l_self_modules_block10_modules_rep_modules_5_parameters_bias_ = (
            L_self_modules_block10_modules_rep_modules_5_parameters_bias_
        )
        l_self_modules_block10_modules_rep_modules_7_modules_conv1_parameters_weight_ = L_self_modules_block10_modules_rep_modules_7_modules_conv1_parameters_weight_
        l_self_modules_block10_modules_rep_modules_7_modules_pointwise_parameters_weight_ = L_self_modules_block10_modules_rep_modules_7_modules_pointwise_parameters_weight_
        l_self_modules_block10_modules_rep_modules_8_buffers_running_mean_ = (
            L_self_modules_block10_modules_rep_modules_8_buffers_running_mean_
        )
        l_self_modules_block10_modules_rep_modules_8_buffers_running_var_ = (
            L_self_modules_block10_modules_rep_modules_8_buffers_running_var_
        )
        l_self_modules_block10_modules_rep_modules_8_parameters_weight_ = (
            L_self_modules_block10_modules_rep_modules_8_parameters_weight_
        )
        l_self_modules_block10_modules_rep_modules_8_parameters_bias_ = (
            L_self_modules_block10_modules_rep_modules_8_parameters_bias_
        )
        l_self_modules_block11_modules_rep_modules_1_modules_conv1_parameters_weight_ = L_self_modules_block11_modules_rep_modules_1_modules_conv1_parameters_weight_
        l_self_modules_block11_modules_rep_modules_1_modules_pointwise_parameters_weight_ = L_self_modules_block11_modules_rep_modules_1_modules_pointwise_parameters_weight_
        l_self_modules_block11_modules_rep_modules_2_buffers_running_mean_ = (
            L_self_modules_block11_modules_rep_modules_2_buffers_running_mean_
        )
        l_self_modules_block11_modules_rep_modules_2_buffers_running_var_ = (
            L_self_modules_block11_modules_rep_modules_2_buffers_running_var_
        )
        l_self_modules_block11_modules_rep_modules_2_parameters_weight_ = (
            L_self_modules_block11_modules_rep_modules_2_parameters_weight_
        )
        l_self_modules_block11_modules_rep_modules_2_parameters_bias_ = (
            L_self_modules_block11_modules_rep_modules_2_parameters_bias_
        )
        l_self_modules_block11_modules_rep_modules_4_modules_conv1_parameters_weight_ = L_self_modules_block11_modules_rep_modules_4_modules_conv1_parameters_weight_
        l_self_modules_block11_modules_rep_modules_4_modules_pointwise_parameters_weight_ = L_self_modules_block11_modules_rep_modules_4_modules_pointwise_parameters_weight_
        l_self_modules_block11_modules_rep_modules_5_buffers_running_mean_ = (
            L_self_modules_block11_modules_rep_modules_5_buffers_running_mean_
        )
        l_self_modules_block11_modules_rep_modules_5_buffers_running_var_ = (
            L_self_modules_block11_modules_rep_modules_5_buffers_running_var_
        )
        l_self_modules_block11_modules_rep_modules_5_parameters_weight_ = (
            L_self_modules_block11_modules_rep_modules_5_parameters_weight_
        )
        l_self_modules_block11_modules_rep_modules_5_parameters_bias_ = (
            L_self_modules_block11_modules_rep_modules_5_parameters_bias_
        )
        l_self_modules_block11_modules_rep_modules_7_modules_conv1_parameters_weight_ = L_self_modules_block11_modules_rep_modules_7_modules_conv1_parameters_weight_
        l_self_modules_block11_modules_rep_modules_7_modules_pointwise_parameters_weight_ = L_self_modules_block11_modules_rep_modules_7_modules_pointwise_parameters_weight_
        l_self_modules_block11_modules_rep_modules_8_buffers_running_mean_ = (
            L_self_modules_block11_modules_rep_modules_8_buffers_running_mean_
        )
        l_self_modules_block11_modules_rep_modules_8_buffers_running_var_ = (
            L_self_modules_block11_modules_rep_modules_8_buffers_running_var_
        )
        l_self_modules_block11_modules_rep_modules_8_parameters_weight_ = (
            L_self_modules_block11_modules_rep_modules_8_parameters_weight_
        )
        l_self_modules_block11_modules_rep_modules_8_parameters_bias_ = (
            L_self_modules_block11_modules_rep_modules_8_parameters_bias_
        )
        l_self_modules_block12_modules_rep_modules_1_modules_conv1_parameters_weight_ = L_self_modules_block12_modules_rep_modules_1_modules_conv1_parameters_weight_
        l_self_modules_block12_modules_rep_modules_1_modules_pointwise_parameters_weight_ = L_self_modules_block12_modules_rep_modules_1_modules_pointwise_parameters_weight_
        l_self_modules_block12_modules_rep_modules_2_buffers_running_mean_ = (
            L_self_modules_block12_modules_rep_modules_2_buffers_running_mean_
        )
        l_self_modules_block12_modules_rep_modules_2_buffers_running_var_ = (
            L_self_modules_block12_modules_rep_modules_2_buffers_running_var_
        )
        l_self_modules_block12_modules_rep_modules_2_parameters_weight_ = (
            L_self_modules_block12_modules_rep_modules_2_parameters_weight_
        )
        l_self_modules_block12_modules_rep_modules_2_parameters_bias_ = (
            L_self_modules_block12_modules_rep_modules_2_parameters_bias_
        )
        l_self_modules_block12_modules_rep_modules_4_modules_conv1_parameters_weight_ = L_self_modules_block12_modules_rep_modules_4_modules_conv1_parameters_weight_
        l_self_modules_block12_modules_rep_modules_4_modules_pointwise_parameters_weight_ = L_self_modules_block12_modules_rep_modules_4_modules_pointwise_parameters_weight_
        l_self_modules_block12_modules_rep_modules_5_buffers_running_mean_ = (
            L_self_modules_block12_modules_rep_modules_5_buffers_running_mean_
        )
        l_self_modules_block12_modules_rep_modules_5_buffers_running_var_ = (
            L_self_modules_block12_modules_rep_modules_5_buffers_running_var_
        )
        l_self_modules_block12_modules_rep_modules_5_parameters_weight_ = (
            L_self_modules_block12_modules_rep_modules_5_parameters_weight_
        )
        l_self_modules_block12_modules_rep_modules_5_parameters_bias_ = (
            L_self_modules_block12_modules_rep_modules_5_parameters_bias_
        )
        l_self_modules_block12_modules_skip_parameters_weight_ = (
            L_self_modules_block12_modules_skip_parameters_weight_
        )
        l_self_modules_block12_modules_skipbn_buffers_running_mean_ = (
            L_self_modules_block12_modules_skipbn_buffers_running_mean_
        )
        l_self_modules_block12_modules_skipbn_buffers_running_var_ = (
            L_self_modules_block12_modules_skipbn_buffers_running_var_
        )
        l_self_modules_block12_modules_skipbn_parameters_weight_ = (
            L_self_modules_block12_modules_skipbn_parameters_weight_
        )
        l_self_modules_block12_modules_skipbn_parameters_bias_ = (
            L_self_modules_block12_modules_skipbn_parameters_bias_
        )
        l_self_modules_conv3_modules_conv1_parameters_weight_ = (
            L_self_modules_conv3_modules_conv1_parameters_weight_
        )
        l_self_modules_conv3_modules_pointwise_parameters_weight_ = (
            L_self_modules_conv3_modules_pointwise_parameters_weight_
        )
        l_self_modules_bn3_buffers_running_mean_ = (
            L_self_modules_bn3_buffers_running_mean_
        )
        l_self_modules_bn3_buffers_running_var_ = (
            L_self_modules_bn3_buffers_running_var_
        )
        l_self_modules_bn3_parameters_weight_ = L_self_modules_bn3_parameters_weight_
        l_self_modules_bn3_parameters_bias_ = L_self_modules_bn3_parameters_bias_
        l_self_modules_conv4_modules_conv1_parameters_weight_ = (
            L_self_modules_conv4_modules_conv1_parameters_weight_
        )
        l_self_modules_conv4_modules_pointwise_parameters_weight_ = (
            L_self_modules_conv4_modules_pointwise_parameters_weight_
        )
        l_self_modules_bn4_buffers_running_mean_ = (
            L_self_modules_bn4_buffers_running_mean_
        )
        l_self_modules_bn4_buffers_running_var_ = (
            L_self_modules_bn4_buffers_running_var_
        )
        l_self_modules_bn4_parameters_weight_ = L_self_modules_bn4_parameters_weight_
        l_self_modules_bn4_parameters_bias_ = L_self_modules_bn4_parameters_bias_
        l_self_modules_fc_parameters_weight_ = L_self_modules_fc_parameters_weight_
        l_self_modules_fc_parameters_bias_ = L_self_modules_fc_parameters_bias_
        x = torch.conv2d(
            l_x_,
            l_self_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_conv1_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_bn1_buffers_running_mean_,
            l_self_modules_bn1_buffers_running_var_,
            l_self_modules_bn1_parameters_weight_,
            l_self_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_bn1_parameters_weight_
        ) = l_self_modules_bn1_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_conv2_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_bn2_buffers_running_mean_,
            l_self_modules_bn2_buffers_running_var_,
            l_self_modules_bn2_parameters_weight_,
            l_self_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_bn2_parameters_weight_
        ) = l_self_modules_bn2_parameters_bias_ = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_block1_modules_rep_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_block1_modules_rep_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_block1_modules_rep_modules_0_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_block1_modules_rep_modules_0_modules_pointwise_parameters_weight_ = (None)
        input_1 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_block1_modules_rep_modules_1_buffers_running_mean_,
            l_self_modules_block1_modules_rep_modules_1_buffers_running_var_,
            l_self_modules_block1_modules_rep_modules_1_parameters_weight_,
            l_self_modules_block1_modules_rep_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = (
            l_self_modules_block1_modules_rep_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_block1_modules_rep_modules_1_buffers_running_var_
        ) = (
            l_self_modules_block1_modules_rep_modules_1_parameters_weight_
        ) = l_self_modules_block1_modules_rep_modules_1_parameters_bias_ = None
        input_2 = torch.nn.functional.relu(input_1, inplace=True)
        input_1 = None
        x_8 = torch.conv2d(
            input_2,
            l_self_modules_block1_modules_rep_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_2 = (
            l_self_modules_block1_modules_rep_modules_3_modules_conv1_parameters_weight_
        ) = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_block1_modules_rep_modules_3_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_block1_modules_rep_modules_3_modules_pointwise_parameters_weight_ = (None)
        input_3 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_block1_modules_rep_modules_4_buffers_running_mean_,
            l_self_modules_block1_modules_rep_modules_4_buffers_running_var_,
            l_self_modules_block1_modules_rep_modules_4_parameters_weight_,
            l_self_modules_block1_modules_rep_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = (
            l_self_modules_block1_modules_rep_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_block1_modules_rep_modules_4_buffers_running_var_
        ) = (
            l_self_modules_block1_modules_rep_modules_4_parameters_weight_
        ) = l_self_modules_block1_modules_rep_modules_4_parameters_bias_ = None
        input_4 = torch.nn.functional.max_pool2d(
            input_3, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        input_3 = None
        skip = torch.conv2d(
            x_5,
            l_self_modules_block1_modules_skip_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_block1_modules_skip_parameters_weight_ = None
        skip_1 = torch.nn.functional.batch_norm(
            skip,
            l_self_modules_block1_modules_skipbn_buffers_running_mean_,
            l_self_modules_block1_modules_skipbn_buffers_running_var_,
            l_self_modules_block1_modules_skipbn_parameters_weight_,
            l_self_modules_block1_modules_skipbn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        skip = (
            l_self_modules_block1_modules_skipbn_buffers_running_mean_
        ) = (
            l_self_modules_block1_modules_skipbn_buffers_running_var_
        ) = (
            l_self_modules_block1_modules_skipbn_parameters_weight_
        ) = l_self_modules_block1_modules_skipbn_parameters_bias_ = None
        input_4 += skip_1
        x_10 = input_4
        input_4 = skip_1 = None
        input_5 = torch.nn.functional.relu(x_10, inplace=False)
        x_11 = torch.conv2d(
            input_5,
            l_self_modules_block2_modules_rep_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_5 = (
            l_self_modules_block2_modules_rep_modules_1_modules_conv1_parameters_weight_
        ) = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_block2_modules_rep_modules_1_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_block2_modules_rep_modules_1_modules_pointwise_parameters_weight_ = (None)
        input_6 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_block2_modules_rep_modules_2_buffers_running_mean_,
            l_self_modules_block2_modules_rep_modules_2_buffers_running_var_,
            l_self_modules_block2_modules_rep_modules_2_parameters_weight_,
            l_self_modules_block2_modules_rep_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = (
            l_self_modules_block2_modules_rep_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_block2_modules_rep_modules_2_buffers_running_var_
        ) = (
            l_self_modules_block2_modules_rep_modules_2_parameters_weight_
        ) = l_self_modules_block2_modules_rep_modules_2_parameters_bias_ = None
        input_7 = torch.nn.functional.relu(input_6, inplace=True)
        input_6 = None
        x_13 = torch.conv2d(
            input_7,
            l_self_modules_block2_modules_rep_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_7 = (
            l_self_modules_block2_modules_rep_modules_4_modules_conv1_parameters_weight_
        ) = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_block2_modules_rep_modules_4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_block2_modules_rep_modules_4_modules_pointwise_parameters_weight_ = (None)
        input_8 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_block2_modules_rep_modules_5_buffers_running_mean_,
            l_self_modules_block2_modules_rep_modules_5_buffers_running_var_,
            l_self_modules_block2_modules_rep_modules_5_parameters_weight_,
            l_self_modules_block2_modules_rep_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = (
            l_self_modules_block2_modules_rep_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_block2_modules_rep_modules_5_buffers_running_var_
        ) = (
            l_self_modules_block2_modules_rep_modules_5_parameters_weight_
        ) = l_self_modules_block2_modules_rep_modules_5_parameters_bias_ = None
        input_9 = torch.nn.functional.max_pool2d(
            input_8, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        input_8 = None
        skip_2 = torch.conv2d(
            x_10,
            l_self_modules_block2_modules_skip_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_block2_modules_skip_parameters_weight_ = None
        skip_3 = torch.nn.functional.batch_norm(
            skip_2,
            l_self_modules_block2_modules_skipbn_buffers_running_mean_,
            l_self_modules_block2_modules_skipbn_buffers_running_var_,
            l_self_modules_block2_modules_skipbn_parameters_weight_,
            l_self_modules_block2_modules_skipbn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        skip_2 = (
            l_self_modules_block2_modules_skipbn_buffers_running_mean_
        ) = (
            l_self_modules_block2_modules_skipbn_buffers_running_var_
        ) = (
            l_self_modules_block2_modules_skipbn_parameters_weight_
        ) = l_self_modules_block2_modules_skipbn_parameters_bias_ = None
        input_9 += skip_3
        x_15 = input_9
        input_9 = skip_3 = None
        input_10 = torch.nn.functional.relu(x_15, inplace=False)
        x_16 = torch.conv2d(
            input_10,
            l_self_modules_block3_modules_rep_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_10 = (
            l_self_modules_block3_modules_rep_modules_1_modules_conv1_parameters_weight_
        ) = None
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_block3_modules_rep_modules_1_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_block3_modules_rep_modules_1_modules_pointwise_parameters_weight_ = (None)
        input_11 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_block3_modules_rep_modules_2_buffers_running_mean_,
            l_self_modules_block3_modules_rep_modules_2_buffers_running_var_,
            l_self_modules_block3_modules_rep_modules_2_parameters_weight_,
            l_self_modules_block3_modules_rep_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = (
            l_self_modules_block3_modules_rep_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_block3_modules_rep_modules_2_buffers_running_var_
        ) = (
            l_self_modules_block3_modules_rep_modules_2_parameters_weight_
        ) = l_self_modules_block3_modules_rep_modules_2_parameters_bias_ = None
        input_12 = torch.nn.functional.relu(input_11, inplace=True)
        input_11 = None
        x_18 = torch.conv2d(
            input_12,
            l_self_modules_block3_modules_rep_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_12 = (
            l_self_modules_block3_modules_rep_modules_4_modules_conv1_parameters_weight_
        ) = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_block3_modules_rep_modules_4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_block3_modules_rep_modules_4_modules_pointwise_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_block3_modules_rep_modules_5_buffers_running_mean_,
            l_self_modules_block3_modules_rep_modules_5_buffers_running_var_,
            l_self_modules_block3_modules_rep_modules_5_parameters_weight_,
            l_self_modules_block3_modules_rep_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = (
            l_self_modules_block3_modules_rep_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_block3_modules_rep_modules_5_buffers_running_var_
        ) = (
            l_self_modules_block3_modules_rep_modules_5_parameters_weight_
        ) = l_self_modules_block3_modules_rep_modules_5_parameters_bias_ = None
        input_14 = torch.nn.functional.max_pool2d(
            input_13, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        input_13 = None
        skip_4 = torch.conv2d(
            x_15,
            l_self_modules_block3_modules_skip_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_block3_modules_skip_parameters_weight_ = None
        skip_5 = torch.nn.functional.batch_norm(
            skip_4,
            l_self_modules_block3_modules_skipbn_buffers_running_mean_,
            l_self_modules_block3_modules_skipbn_buffers_running_var_,
            l_self_modules_block3_modules_skipbn_parameters_weight_,
            l_self_modules_block3_modules_skipbn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        skip_4 = (
            l_self_modules_block3_modules_skipbn_buffers_running_mean_
        ) = (
            l_self_modules_block3_modules_skipbn_buffers_running_var_
        ) = (
            l_self_modules_block3_modules_skipbn_parameters_weight_
        ) = l_self_modules_block3_modules_skipbn_parameters_bias_ = None
        input_14 += skip_5
        x_20 = input_14
        input_14 = skip_5 = None
        input_15 = torch.nn.functional.relu(x_20, inplace=False)
        x_21 = torch.conv2d(
            input_15,
            l_self_modules_block4_modules_rep_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_15 = (
            l_self_modules_block4_modules_rep_modules_1_modules_conv1_parameters_weight_
        ) = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_block4_modules_rep_modules_1_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_block4_modules_rep_modules_1_modules_pointwise_parameters_weight_ = (None)
        input_16 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_block4_modules_rep_modules_2_buffers_running_mean_,
            l_self_modules_block4_modules_rep_modules_2_buffers_running_var_,
            l_self_modules_block4_modules_rep_modules_2_parameters_weight_,
            l_self_modules_block4_modules_rep_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = (
            l_self_modules_block4_modules_rep_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_block4_modules_rep_modules_2_buffers_running_var_
        ) = (
            l_self_modules_block4_modules_rep_modules_2_parameters_weight_
        ) = l_self_modules_block4_modules_rep_modules_2_parameters_bias_ = None
        input_17 = torch.nn.functional.relu(input_16, inplace=True)
        input_16 = None
        x_23 = torch.conv2d(
            input_17,
            l_self_modules_block4_modules_rep_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_17 = (
            l_self_modules_block4_modules_rep_modules_4_modules_conv1_parameters_weight_
        ) = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_block4_modules_rep_modules_4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_block4_modules_rep_modules_4_modules_pointwise_parameters_weight_ = (None)
        input_18 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_block4_modules_rep_modules_5_buffers_running_mean_,
            l_self_modules_block4_modules_rep_modules_5_buffers_running_var_,
            l_self_modules_block4_modules_rep_modules_5_parameters_weight_,
            l_self_modules_block4_modules_rep_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = (
            l_self_modules_block4_modules_rep_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_block4_modules_rep_modules_5_buffers_running_var_
        ) = (
            l_self_modules_block4_modules_rep_modules_5_parameters_weight_
        ) = l_self_modules_block4_modules_rep_modules_5_parameters_bias_ = None
        input_19 = torch.nn.functional.relu(input_18, inplace=True)
        input_18 = None
        x_25 = torch.conv2d(
            input_19,
            l_self_modules_block4_modules_rep_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_19 = (
            l_self_modules_block4_modules_rep_modules_7_modules_conv1_parameters_weight_
        ) = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_block4_modules_rep_modules_7_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_block4_modules_rep_modules_7_modules_pointwise_parameters_weight_ = (None)
        input_20 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_block4_modules_rep_modules_8_buffers_running_mean_,
            l_self_modules_block4_modules_rep_modules_8_buffers_running_var_,
            l_self_modules_block4_modules_rep_modules_8_parameters_weight_,
            l_self_modules_block4_modules_rep_modules_8_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = (
            l_self_modules_block4_modules_rep_modules_8_buffers_running_mean_
        ) = (
            l_self_modules_block4_modules_rep_modules_8_buffers_running_var_
        ) = (
            l_self_modules_block4_modules_rep_modules_8_parameters_weight_
        ) = l_self_modules_block4_modules_rep_modules_8_parameters_bias_ = None
        input_20 += x_20
        x_27 = input_20
        input_20 = x_20 = None
        input_21 = torch.nn.functional.relu(x_27, inplace=False)
        x_28 = torch.conv2d(
            input_21,
            l_self_modules_block5_modules_rep_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_21 = (
            l_self_modules_block5_modules_rep_modules_1_modules_conv1_parameters_weight_
        ) = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_block5_modules_rep_modules_1_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_block5_modules_rep_modules_1_modules_pointwise_parameters_weight_ = (None)
        input_22 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_block5_modules_rep_modules_2_buffers_running_mean_,
            l_self_modules_block5_modules_rep_modules_2_buffers_running_var_,
            l_self_modules_block5_modules_rep_modules_2_parameters_weight_,
            l_self_modules_block5_modules_rep_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = (
            l_self_modules_block5_modules_rep_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_block5_modules_rep_modules_2_buffers_running_var_
        ) = (
            l_self_modules_block5_modules_rep_modules_2_parameters_weight_
        ) = l_self_modules_block5_modules_rep_modules_2_parameters_bias_ = None
        input_23 = torch.nn.functional.relu(input_22, inplace=True)
        input_22 = None
        x_30 = torch.conv2d(
            input_23,
            l_self_modules_block5_modules_rep_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_23 = (
            l_self_modules_block5_modules_rep_modules_4_modules_conv1_parameters_weight_
        ) = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_block5_modules_rep_modules_4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_block5_modules_rep_modules_4_modules_pointwise_parameters_weight_ = (None)
        input_24 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_block5_modules_rep_modules_5_buffers_running_mean_,
            l_self_modules_block5_modules_rep_modules_5_buffers_running_var_,
            l_self_modules_block5_modules_rep_modules_5_parameters_weight_,
            l_self_modules_block5_modules_rep_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = (
            l_self_modules_block5_modules_rep_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_block5_modules_rep_modules_5_buffers_running_var_
        ) = (
            l_self_modules_block5_modules_rep_modules_5_parameters_weight_
        ) = l_self_modules_block5_modules_rep_modules_5_parameters_bias_ = None
        input_25 = torch.nn.functional.relu(input_24, inplace=True)
        input_24 = None
        x_32 = torch.conv2d(
            input_25,
            l_self_modules_block5_modules_rep_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_25 = (
            l_self_modules_block5_modules_rep_modules_7_modules_conv1_parameters_weight_
        ) = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_block5_modules_rep_modules_7_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_block5_modules_rep_modules_7_modules_pointwise_parameters_weight_ = (None)
        input_26 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_block5_modules_rep_modules_8_buffers_running_mean_,
            l_self_modules_block5_modules_rep_modules_8_buffers_running_var_,
            l_self_modules_block5_modules_rep_modules_8_parameters_weight_,
            l_self_modules_block5_modules_rep_modules_8_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = (
            l_self_modules_block5_modules_rep_modules_8_buffers_running_mean_
        ) = (
            l_self_modules_block5_modules_rep_modules_8_buffers_running_var_
        ) = (
            l_self_modules_block5_modules_rep_modules_8_parameters_weight_
        ) = l_self_modules_block5_modules_rep_modules_8_parameters_bias_ = None
        input_26 += x_27
        x_34 = input_26
        input_26 = x_27 = None
        input_27 = torch.nn.functional.relu(x_34, inplace=False)
        x_35 = torch.conv2d(
            input_27,
            l_self_modules_block6_modules_rep_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_27 = (
            l_self_modules_block6_modules_rep_modules_1_modules_conv1_parameters_weight_
        ) = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_block6_modules_rep_modules_1_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_block6_modules_rep_modules_1_modules_pointwise_parameters_weight_ = (None)
        input_28 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_block6_modules_rep_modules_2_buffers_running_mean_,
            l_self_modules_block6_modules_rep_modules_2_buffers_running_var_,
            l_self_modules_block6_modules_rep_modules_2_parameters_weight_,
            l_self_modules_block6_modules_rep_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = (
            l_self_modules_block6_modules_rep_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_block6_modules_rep_modules_2_buffers_running_var_
        ) = (
            l_self_modules_block6_modules_rep_modules_2_parameters_weight_
        ) = l_self_modules_block6_modules_rep_modules_2_parameters_bias_ = None
        input_29 = torch.nn.functional.relu(input_28, inplace=True)
        input_28 = None
        x_37 = torch.conv2d(
            input_29,
            l_self_modules_block6_modules_rep_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_29 = (
            l_self_modules_block6_modules_rep_modules_4_modules_conv1_parameters_weight_
        ) = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_block6_modules_rep_modules_4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_block6_modules_rep_modules_4_modules_pointwise_parameters_weight_ = (None)
        input_30 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_block6_modules_rep_modules_5_buffers_running_mean_,
            l_self_modules_block6_modules_rep_modules_5_buffers_running_var_,
            l_self_modules_block6_modules_rep_modules_5_parameters_weight_,
            l_self_modules_block6_modules_rep_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = (
            l_self_modules_block6_modules_rep_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_block6_modules_rep_modules_5_buffers_running_var_
        ) = (
            l_self_modules_block6_modules_rep_modules_5_parameters_weight_
        ) = l_self_modules_block6_modules_rep_modules_5_parameters_bias_ = None
        input_31 = torch.nn.functional.relu(input_30, inplace=True)
        input_30 = None
        x_39 = torch.conv2d(
            input_31,
            l_self_modules_block6_modules_rep_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_31 = (
            l_self_modules_block6_modules_rep_modules_7_modules_conv1_parameters_weight_
        ) = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_block6_modules_rep_modules_7_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_block6_modules_rep_modules_7_modules_pointwise_parameters_weight_ = (None)
        input_32 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_block6_modules_rep_modules_8_buffers_running_mean_,
            l_self_modules_block6_modules_rep_modules_8_buffers_running_var_,
            l_self_modules_block6_modules_rep_modules_8_parameters_weight_,
            l_self_modules_block6_modules_rep_modules_8_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = (
            l_self_modules_block6_modules_rep_modules_8_buffers_running_mean_
        ) = (
            l_self_modules_block6_modules_rep_modules_8_buffers_running_var_
        ) = (
            l_self_modules_block6_modules_rep_modules_8_parameters_weight_
        ) = l_self_modules_block6_modules_rep_modules_8_parameters_bias_ = None
        input_32 += x_34
        x_41 = input_32
        input_32 = x_34 = None
        input_33 = torch.nn.functional.relu(x_41, inplace=False)
        x_42 = torch.conv2d(
            input_33,
            l_self_modules_block7_modules_rep_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_33 = (
            l_self_modules_block7_modules_rep_modules_1_modules_conv1_parameters_weight_
        ) = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_block7_modules_rep_modules_1_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_block7_modules_rep_modules_1_modules_pointwise_parameters_weight_ = (None)
        input_34 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_block7_modules_rep_modules_2_buffers_running_mean_,
            l_self_modules_block7_modules_rep_modules_2_buffers_running_var_,
            l_self_modules_block7_modules_rep_modules_2_parameters_weight_,
            l_self_modules_block7_modules_rep_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = (
            l_self_modules_block7_modules_rep_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_block7_modules_rep_modules_2_buffers_running_var_
        ) = (
            l_self_modules_block7_modules_rep_modules_2_parameters_weight_
        ) = l_self_modules_block7_modules_rep_modules_2_parameters_bias_ = None
        input_35 = torch.nn.functional.relu(input_34, inplace=True)
        input_34 = None
        x_44 = torch.conv2d(
            input_35,
            l_self_modules_block7_modules_rep_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_35 = (
            l_self_modules_block7_modules_rep_modules_4_modules_conv1_parameters_weight_
        ) = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_block7_modules_rep_modules_4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_block7_modules_rep_modules_4_modules_pointwise_parameters_weight_ = (None)
        input_36 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_block7_modules_rep_modules_5_buffers_running_mean_,
            l_self_modules_block7_modules_rep_modules_5_buffers_running_var_,
            l_self_modules_block7_modules_rep_modules_5_parameters_weight_,
            l_self_modules_block7_modules_rep_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = (
            l_self_modules_block7_modules_rep_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_block7_modules_rep_modules_5_buffers_running_var_
        ) = (
            l_self_modules_block7_modules_rep_modules_5_parameters_weight_
        ) = l_self_modules_block7_modules_rep_modules_5_parameters_bias_ = None
        input_37 = torch.nn.functional.relu(input_36, inplace=True)
        input_36 = None
        x_46 = torch.conv2d(
            input_37,
            l_self_modules_block7_modules_rep_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_37 = (
            l_self_modules_block7_modules_rep_modules_7_modules_conv1_parameters_weight_
        ) = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_block7_modules_rep_modules_7_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_block7_modules_rep_modules_7_modules_pointwise_parameters_weight_ = (None)
        input_38 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_block7_modules_rep_modules_8_buffers_running_mean_,
            l_self_modules_block7_modules_rep_modules_8_buffers_running_var_,
            l_self_modules_block7_modules_rep_modules_8_parameters_weight_,
            l_self_modules_block7_modules_rep_modules_8_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = (
            l_self_modules_block7_modules_rep_modules_8_buffers_running_mean_
        ) = (
            l_self_modules_block7_modules_rep_modules_8_buffers_running_var_
        ) = (
            l_self_modules_block7_modules_rep_modules_8_parameters_weight_
        ) = l_self_modules_block7_modules_rep_modules_8_parameters_bias_ = None
        input_38 += x_41
        x_48 = input_38
        input_38 = x_41 = None
        input_39 = torch.nn.functional.relu(x_48, inplace=False)
        x_49 = torch.conv2d(
            input_39,
            l_self_modules_block8_modules_rep_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_39 = (
            l_self_modules_block8_modules_rep_modules_1_modules_conv1_parameters_weight_
        ) = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_block8_modules_rep_modules_1_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_block8_modules_rep_modules_1_modules_pointwise_parameters_weight_ = (None)
        input_40 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_block8_modules_rep_modules_2_buffers_running_mean_,
            l_self_modules_block8_modules_rep_modules_2_buffers_running_var_,
            l_self_modules_block8_modules_rep_modules_2_parameters_weight_,
            l_self_modules_block8_modules_rep_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = (
            l_self_modules_block8_modules_rep_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_block8_modules_rep_modules_2_buffers_running_var_
        ) = (
            l_self_modules_block8_modules_rep_modules_2_parameters_weight_
        ) = l_self_modules_block8_modules_rep_modules_2_parameters_bias_ = None
        input_41 = torch.nn.functional.relu(input_40, inplace=True)
        input_40 = None
        x_51 = torch.conv2d(
            input_41,
            l_self_modules_block8_modules_rep_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_41 = (
            l_self_modules_block8_modules_rep_modules_4_modules_conv1_parameters_weight_
        ) = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_block8_modules_rep_modules_4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_block8_modules_rep_modules_4_modules_pointwise_parameters_weight_ = (None)
        input_42 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_block8_modules_rep_modules_5_buffers_running_mean_,
            l_self_modules_block8_modules_rep_modules_5_buffers_running_var_,
            l_self_modules_block8_modules_rep_modules_5_parameters_weight_,
            l_self_modules_block8_modules_rep_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = (
            l_self_modules_block8_modules_rep_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_block8_modules_rep_modules_5_buffers_running_var_
        ) = (
            l_self_modules_block8_modules_rep_modules_5_parameters_weight_
        ) = l_self_modules_block8_modules_rep_modules_5_parameters_bias_ = None
        input_43 = torch.nn.functional.relu(input_42, inplace=True)
        input_42 = None
        x_53 = torch.conv2d(
            input_43,
            l_self_modules_block8_modules_rep_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_43 = (
            l_self_modules_block8_modules_rep_modules_7_modules_conv1_parameters_weight_
        ) = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_block8_modules_rep_modules_7_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_block8_modules_rep_modules_7_modules_pointwise_parameters_weight_ = (None)
        input_44 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_block8_modules_rep_modules_8_buffers_running_mean_,
            l_self_modules_block8_modules_rep_modules_8_buffers_running_var_,
            l_self_modules_block8_modules_rep_modules_8_parameters_weight_,
            l_self_modules_block8_modules_rep_modules_8_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = (
            l_self_modules_block8_modules_rep_modules_8_buffers_running_mean_
        ) = (
            l_self_modules_block8_modules_rep_modules_8_buffers_running_var_
        ) = (
            l_self_modules_block8_modules_rep_modules_8_parameters_weight_
        ) = l_self_modules_block8_modules_rep_modules_8_parameters_bias_ = None
        input_44 += x_48
        x_55 = input_44
        input_44 = x_48 = None
        input_45 = torch.nn.functional.relu(x_55, inplace=False)
        x_56 = torch.conv2d(
            input_45,
            l_self_modules_block9_modules_rep_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_45 = (
            l_self_modules_block9_modules_rep_modules_1_modules_conv1_parameters_weight_
        ) = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_block9_modules_rep_modules_1_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_56 = l_self_modules_block9_modules_rep_modules_1_modules_pointwise_parameters_weight_ = (None)
        input_46 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_block9_modules_rep_modules_2_buffers_running_mean_,
            l_self_modules_block9_modules_rep_modules_2_buffers_running_var_,
            l_self_modules_block9_modules_rep_modules_2_parameters_weight_,
            l_self_modules_block9_modules_rep_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = (
            l_self_modules_block9_modules_rep_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_block9_modules_rep_modules_2_buffers_running_var_
        ) = (
            l_self_modules_block9_modules_rep_modules_2_parameters_weight_
        ) = l_self_modules_block9_modules_rep_modules_2_parameters_bias_ = None
        input_47 = torch.nn.functional.relu(input_46, inplace=True)
        input_46 = None
        x_58 = torch.conv2d(
            input_47,
            l_self_modules_block9_modules_rep_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_47 = (
            l_self_modules_block9_modules_rep_modules_4_modules_conv1_parameters_weight_
        ) = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_block9_modules_rep_modules_4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_block9_modules_rep_modules_4_modules_pointwise_parameters_weight_ = (None)
        input_48 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_block9_modules_rep_modules_5_buffers_running_mean_,
            l_self_modules_block9_modules_rep_modules_5_buffers_running_var_,
            l_self_modules_block9_modules_rep_modules_5_parameters_weight_,
            l_self_modules_block9_modules_rep_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = (
            l_self_modules_block9_modules_rep_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_block9_modules_rep_modules_5_buffers_running_var_
        ) = (
            l_self_modules_block9_modules_rep_modules_5_parameters_weight_
        ) = l_self_modules_block9_modules_rep_modules_5_parameters_bias_ = None
        input_49 = torch.nn.functional.relu(input_48, inplace=True)
        input_48 = None
        x_60 = torch.conv2d(
            input_49,
            l_self_modules_block9_modules_rep_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_49 = (
            l_self_modules_block9_modules_rep_modules_7_modules_conv1_parameters_weight_
        ) = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_block9_modules_rep_modules_7_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_block9_modules_rep_modules_7_modules_pointwise_parameters_weight_ = (None)
        input_50 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_block9_modules_rep_modules_8_buffers_running_mean_,
            l_self_modules_block9_modules_rep_modules_8_buffers_running_var_,
            l_self_modules_block9_modules_rep_modules_8_parameters_weight_,
            l_self_modules_block9_modules_rep_modules_8_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = (
            l_self_modules_block9_modules_rep_modules_8_buffers_running_mean_
        ) = (
            l_self_modules_block9_modules_rep_modules_8_buffers_running_var_
        ) = (
            l_self_modules_block9_modules_rep_modules_8_parameters_weight_
        ) = l_self_modules_block9_modules_rep_modules_8_parameters_bias_ = None
        input_50 += x_55
        x_62 = input_50
        input_50 = x_55 = None
        input_51 = torch.nn.functional.relu(x_62, inplace=False)
        x_63 = torch.conv2d(
            input_51,
            l_self_modules_block10_modules_rep_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_51 = l_self_modules_block10_modules_rep_modules_1_modules_conv1_parameters_weight_ = (None)
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_block10_modules_rep_modules_1_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_block10_modules_rep_modules_1_modules_pointwise_parameters_weight_ = (None)
        input_52 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_block10_modules_rep_modules_2_buffers_running_mean_,
            l_self_modules_block10_modules_rep_modules_2_buffers_running_var_,
            l_self_modules_block10_modules_rep_modules_2_parameters_weight_,
            l_self_modules_block10_modules_rep_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = (
            l_self_modules_block10_modules_rep_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_block10_modules_rep_modules_2_buffers_running_var_
        ) = (
            l_self_modules_block10_modules_rep_modules_2_parameters_weight_
        ) = l_self_modules_block10_modules_rep_modules_2_parameters_bias_ = None
        input_53 = torch.nn.functional.relu(input_52, inplace=True)
        input_52 = None
        x_65 = torch.conv2d(
            input_53,
            l_self_modules_block10_modules_rep_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_53 = l_self_modules_block10_modules_rep_modules_4_modules_conv1_parameters_weight_ = (None)
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_block10_modules_rep_modules_4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_block10_modules_rep_modules_4_modules_pointwise_parameters_weight_ = (None)
        input_54 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_block10_modules_rep_modules_5_buffers_running_mean_,
            l_self_modules_block10_modules_rep_modules_5_buffers_running_var_,
            l_self_modules_block10_modules_rep_modules_5_parameters_weight_,
            l_self_modules_block10_modules_rep_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = (
            l_self_modules_block10_modules_rep_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_block10_modules_rep_modules_5_buffers_running_var_
        ) = (
            l_self_modules_block10_modules_rep_modules_5_parameters_weight_
        ) = l_self_modules_block10_modules_rep_modules_5_parameters_bias_ = None
        input_55 = torch.nn.functional.relu(input_54, inplace=True)
        input_54 = None
        x_67 = torch.conv2d(
            input_55,
            l_self_modules_block10_modules_rep_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_55 = l_self_modules_block10_modules_rep_modules_7_modules_conv1_parameters_weight_ = (None)
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_block10_modules_rep_modules_7_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_block10_modules_rep_modules_7_modules_pointwise_parameters_weight_ = (None)
        input_56 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_block10_modules_rep_modules_8_buffers_running_mean_,
            l_self_modules_block10_modules_rep_modules_8_buffers_running_var_,
            l_self_modules_block10_modules_rep_modules_8_parameters_weight_,
            l_self_modules_block10_modules_rep_modules_8_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = (
            l_self_modules_block10_modules_rep_modules_8_buffers_running_mean_
        ) = (
            l_self_modules_block10_modules_rep_modules_8_buffers_running_var_
        ) = (
            l_self_modules_block10_modules_rep_modules_8_parameters_weight_
        ) = l_self_modules_block10_modules_rep_modules_8_parameters_bias_ = None
        input_56 += x_62
        x_69 = input_56
        input_56 = x_62 = None
        input_57 = torch.nn.functional.relu(x_69, inplace=False)
        x_70 = torch.conv2d(
            input_57,
            l_self_modules_block11_modules_rep_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_57 = l_self_modules_block11_modules_rep_modules_1_modules_conv1_parameters_weight_ = (None)
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_block11_modules_rep_modules_1_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_block11_modules_rep_modules_1_modules_pointwise_parameters_weight_ = (None)
        input_58 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_block11_modules_rep_modules_2_buffers_running_mean_,
            l_self_modules_block11_modules_rep_modules_2_buffers_running_var_,
            l_self_modules_block11_modules_rep_modules_2_parameters_weight_,
            l_self_modules_block11_modules_rep_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = (
            l_self_modules_block11_modules_rep_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_block11_modules_rep_modules_2_buffers_running_var_
        ) = (
            l_self_modules_block11_modules_rep_modules_2_parameters_weight_
        ) = l_self_modules_block11_modules_rep_modules_2_parameters_bias_ = None
        input_59 = torch.nn.functional.relu(input_58, inplace=True)
        input_58 = None
        x_72 = torch.conv2d(
            input_59,
            l_self_modules_block11_modules_rep_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_59 = l_self_modules_block11_modules_rep_modules_4_modules_conv1_parameters_weight_ = (None)
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_block11_modules_rep_modules_4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_block11_modules_rep_modules_4_modules_pointwise_parameters_weight_ = (None)
        input_60 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_block11_modules_rep_modules_5_buffers_running_mean_,
            l_self_modules_block11_modules_rep_modules_5_buffers_running_var_,
            l_self_modules_block11_modules_rep_modules_5_parameters_weight_,
            l_self_modules_block11_modules_rep_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = (
            l_self_modules_block11_modules_rep_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_block11_modules_rep_modules_5_buffers_running_var_
        ) = (
            l_self_modules_block11_modules_rep_modules_5_parameters_weight_
        ) = l_self_modules_block11_modules_rep_modules_5_parameters_bias_ = None
        input_61 = torch.nn.functional.relu(input_60, inplace=True)
        input_60 = None
        x_74 = torch.conv2d(
            input_61,
            l_self_modules_block11_modules_rep_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_61 = l_self_modules_block11_modules_rep_modules_7_modules_conv1_parameters_weight_ = (None)
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_block11_modules_rep_modules_7_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_block11_modules_rep_modules_7_modules_pointwise_parameters_weight_ = (None)
        input_62 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_block11_modules_rep_modules_8_buffers_running_mean_,
            l_self_modules_block11_modules_rep_modules_8_buffers_running_var_,
            l_self_modules_block11_modules_rep_modules_8_parameters_weight_,
            l_self_modules_block11_modules_rep_modules_8_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = (
            l_self_modules_block11_modules_rep_modules_8_buffers_running_mean_
        ) = (
            l_self_modules_block11_modules_rep_modules_8_buffers_running_var_
        ) = (
            l_self_modules_block11_modules_rep_modules_8_parameters_weight_
        ) = l_self_modules_block11_modules_rep_modules_8_parameters_bias_ = None
        input_62 += x_69
        x_76 = input_62
        input_62 = x_69 = None
        input_63 = torch.nn.functional.relu(x_76, inplace=False)
        x_77 = torch.conv2d(
            input_63,
            l_self_modules_block12_modules_rep_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_63 = l_self_modules_block12_modules_rep_modules_1_modules_conv1_parameters_weight_ = (None)
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_block12_modules_rep_modules_1_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_77 = l_self_modules_block12_modules_rep_modules_1_modules_pointwise_parameters_weight_ = (None)
        input_64 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_block12_modules_rep_modules_2_buffers_running_mean_,
            l_self_modules_block12_modules_rep_modules_2_buffers_running_var_,
            l_self_modules_block12_modules_rep_modules_2_parameters_weight_,
            l_self_modules_block12_modules_rep_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = (
            l_self_modules_block12_modules_rep_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_block12_modules_rep_modules_2_buffers_running_var_
        ) = (
            l_self_modules_block12_modules_rep_modules_2_parameters_weight_
        ) = l_self_modules_block12_modules_rep_modules_2_parameters_bias_ = None
        input_65 = torch.nn.functional.relu(input_64, inplace=True)
        input_64 = None
        x_79 = torch.conv2d(
            input_65,
            l_self_modules_block12_modules_rep_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        input_65 = l_self_modules_block12_modules_rep_modules_4_modules_conv1_parameters_weight_ = (None)
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_block12_modules_rep_modules_4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_block12_modules_rep_modules_4_modules_pointwise_parameters_weight_ = (None)
        input_66 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_block12_modules_rep_modules_5_buffers_running_mean_,
            l_self_modules_block12_modules_rep_modules_5_buffers_running_var_,
            l_self_modules_block12_modules_rep_modules_5_parameters_weight_,
            l_self_modules_block12_modules_rep_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = (
            l_self_modules_block12_modules_rep_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_block12_modules_rep_modules_5_buffers_running_var_
        ) = (
            l_self_modules_block12_modules_rep_modules_5_parameters_weight_
        ) = l_self_modules_block12_modules_rep_modules_5_parameters_bias_ = None
        input_67 = torch.nn.functional.max_pool2d(
            input_66, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        input_66 = None
        skip_6 = torch.conv2d(
            x_76,
            l_self_modules_block12_modules_skip_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_block12_modules_skip_parameters_weight_ = None
        skip_7 = torch.nn.functional.batch_norm(
            skip_6,
            l_self_modules_block12_modules_skipbn_buffers_running_mean_,
            l_self_modules_block12_modules_skipbn_buffers_running_var_,
            l_self_modules_block12_modules_skipbn_parameters_weight_,
            l_self_modules_block12_modules_skipbn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        skip_6 = (
            l_self_modules_block12_modules_skipbn_buffers_running_mean_
        ) = (
            l_self_modules_block12_modules_skipbn_buffers_running_var_
        ) = (
            l_self_modules_block12_modules_skipbn_parameters_weight_
        ) = l_self_modules_block12_modules_skipbn_parameters_bias_ = None
        input_67 += skip_7
        x_81 = input_67
        input_67 = skip_7 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_conv3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_81 = l_self_modules_conv3_modules_conv1_parameters_weight_ = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_conv3_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_conv3_modules_pointwise_parameters_weight_ = None
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_bn3_buffers_running_mean_,
            l_self_modules_bn3_buffers_running_var_,
            l_self_modules_bn3_parameters_weight_,
            l_self_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = (
            l_self_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_bn3_parameters_weight_
        ) = l_self_modules_bn3_parameters_bias_ = None
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_conv4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_85 = l_self_modules_conv4_modules_conv1_parameters_weight_ = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_conv4_modules_pointwise_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_conv4_modules_pointwise_parameters_weight_ = None
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_bn4_buffers_running_mean_,
            l_self_modules_bn4_buffers_running_var_,
            l_self_modules_bn4_parameters_weight_,
            l_self_modules_bn4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = (
            l_self_modules_bn4_buffers_running_mean_
        ) = (
            l_self_modules_bn4_buffers_running_var_
        ) = (
            l_self_modules_bn4_parameters_weight_
        ) = l_self_modules_bn4_parameters_bias_ = None
        x_89 = torch.nn.functional.relu(x_88, inplace=True)
        x_88 = None
        x_90 = torch.nn.functional.adaptive_avg_pool2d(x_89, 1)
        x_89 = None
        x_91 = x_90.flatten(1, -1)
        x_90 = None
        x_92 = torch._C._nn.linear(
            x_91,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_91 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_92,)
