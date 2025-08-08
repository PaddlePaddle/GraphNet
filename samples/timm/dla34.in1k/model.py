import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_base_layer_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_base_layer_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_base_layer_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_base_layer_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_base_layer_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_level0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_level1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_project_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_project_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level2_modules_project_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_level2_modules_project_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_project_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level2_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level2_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level2_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level2_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level2_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level2_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level2_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level2_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level2_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level2_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_project_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_project_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_project_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_project_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_project_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_project_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_project_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_project_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_project_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_project_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_project_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_project_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level5_modules_project_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_level5_modules_project_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_project_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level5_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level5_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level5_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level5_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level5_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level5_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level5_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level5_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level5_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level5_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_base_layer_modules_0_parameters_weight_ = (
            L_self_modules_base_layer_modules_0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_base_layer_modules_1_buffers_running_mean_ = (
            L_self_modules_base_layer_modules_1_buffers_running_mean_
        )
        l_self_modules_base_layer_modules_1_buffers_running_var_ = (
            L_self_modules_base_layer_modules_1_buffers_running_var_
        )
        l_self_modules_base_layer_modules_1_parameters_weight_ = (
            L_self_modules_base_layer_modules_1_parameters_weight_
        )
        l_self_modules_base_layer_modules_1_parameters_bias_ = (
            L_self_modules_base_layer_modules_1_parameters_bias_
        )
        l_self_modules_level0_modules_0_parameters_weight_ = (
            L_self_modules_level0_modules_0_parameters_weight_
        )
        l_self_modules_level0_modules_1_buffers_running_mean_ = (
            L_self_modules_level0_modules_1_buffers_running_mean_
        )
        l_self_modules_level0_modules_1_buffers_running_var_ = (
            L_self_modules_level0_modules_1_buffers_running_var_
        )
        l_self_modules_level0_modules_1_parameters_weight_ = (
            L_self_modules_level0_modules_1_parameters_weight_
        )
        l_self_modules_level0_modules_1_parameters_bias_ = (
            L_self_modules_level0_modules_1_parameters_bias_
        )
        l_self_modules_level1_modules_0_parameters_weight_ = (
            L_self_modules_level1_modules_0_parameters_weight_
        )
        l_self_modules_level1_modules_1_buffers_running_mean_ = (
            L_self_modules_level1_modules_1_buffers_running_mean_
        )
        l_self_modules_level1_modules_1_buffers_running_var_ = (
            L_self_modules_level1_modules_1_buffers_running_var_
        )
        l_self_modules_level1_modules_1_parameters_weight_ = (
            L_self_modules_level1_modules_1_parameters_weight_
        )
        l_self_modules_level1_modules_1_parameters_bias_ = (
            L_self_modules_level1_modules_1_parameters_bias_
        )
        l_self_modules_level2_modules_project_modules_0_parameters_weight_ = (
            L_self_modules_level2_modules_project_modules_0_parameters_weight_
        )
        l_self_modules_level2_modules_project_modules_1_buffers_running_mean_ = (
            L_self_modules_level2_modules_project_modules_1_buffers_running_mean_
        )
        l_self_modules_level2_modules_project_modules_1_buffers_running_var_ = (
            L_self_modules_level2_modules_project_modules_1_buffers_running_var_
        )
        l_self_modules_level2_modules_project_modules_1_parameters_weight_ = (
            L_self_modules_level2_modules_project_modules_1_parameters_weight_
        )
        l_self_modules_level2_modules_project_modules_1_parameters_bias_ = (
            L_self_modules_level2_modules_project_modules_1_parameters_bias_
        )
        l_self_modules_level2_modules_tree1_modules_conv1_parameters_weight_ = (
            L_self_modules_level2_modules_tree1_modules_conv1_parameters_weight_
        )
        l_self_modules_level2_modules_tree1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_level2_modules_tree1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_level2_modules_tree1_modules_bn1_buffers_running_var_ = (
            L_self_modules_level2_modules_tree1_modules_bn1_buffers_running_var_
        )
        l_self_modules_level2_modules_tree1_modules_bn1_parameters_weight_ = (
            L_self_modules_level2_modules_tree1_modules_bn1_parameters_weight_
        )
        l_self_modules_level2_modules_tree1_modules_bn1_parameters_bias_ = (
            L_self_modules_level2_modules_tree1_modules_bn1_parameters_bias_
        )
        l_self_modules_level2_modules_tree1_modules_conv2_parameters_weight_ = (
            L_self_modules_level2_modules_tree1_modules_conv2_parameters_weight_
        )
        l_self_modules_level2_modules_tree1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_level2_modules_tree1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_level2_modules_tree1_modules_bn2_buffers_running_var_ = (
            L_self_modules_level2_modules_tree1_modules_bn2_buffers_running_var_
        )
        l_self_modules_level2_modules_tree1_modules_bn2_parameters_weight_ = (
            L_self_modules_level2_modules_tree1_modules_bn2_parameters_weight_
        )
        l_self_modules_level2_modules_tree1_modules_bn2_parameters_bias_ = (
            L_self_modules_level2_modules_tree1_modules_bn2_parameters_bias_
        )
        l_self_modules_level2_modules_tree2_modules_conv1_parameters_weight_ = (
            L_self_modules_level2_modules_tree2_modules_conv1_parameters_weight_
        )
        l_self_modules_level2_modules_tree2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_level2_modules_tree2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_level2_modules_tree2_modules_bn1_buffers_running_var_ = (
            L_self_modules_level2_modules_tree2_modules_bn1_buffers_running_var_
        )
        l_self_modules_level2_modules_tree2_modules_bn1_parameters_weight_ = (
            L_self_modules_level2_modules_tree2_modules_bn1_parameters_weight_
        )
        l_self_modules_level2_modules_tree2_modules_bn1_parameters_bias_ = (
            L_self_modules_level2_modules_tree2_modules_bn1_parameters_bias_
        )
        l_self_modules_level2_modules_tree2_modules_conv2_parameters_weight_ = (
            L_self_modules_level2_modules_tree2_modules_conv2_parameters_weight_
        )
        l_self_modules_level2_modules_tree2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_level2_modules_tree2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_level2_modules_tree2_modules_bn2_buffers_running_var_ = (
            L_self_modules_level2_modules_tree2_modules_bn2_buffers_running_var_
        )
        l_self_modules_level2_modules_tree2_modules_bn2_parameters_weight_ = (
            L_self_modules_level2_modules_tree2_modules_bn2_parameters_weight_
        )
        l_self_modules_level2_modules_tree2_modules_bn2_parameters_bias_ = (
            L_self_modules_level2_modules_tree2_modules_bn2_parameters_bias_
        )
        l_self_modules_level2_modules_root_modules_conv_parameters_weight_ = (
            L_self_modules_level2_modules_root_modules_conv_parameters_weight_
        )
        l_self_modules_level2_modules_root_modules_bn_buffers_running_mean_ = (
            L_self_modules_level2_modules_root_modules_bn_buffers_running_mean_
        )
        l_self_modules_level2_modules_root_modules_bn_buffers_running_var_ = (
            L_self_modules_level2_modules_root_modules_bn_buffers_running_var_
        )
        l_self_modules_level2_modules_root_modules_bn_parameters_weight_ = (
            L_self_modules_level2_modules_root_modules_bn_parameters_weight_
        )
        l_self_modules_level2_modules_root_modules_bn_parameters_bias_ = (
            L_self_modules_level2_modules_root_modules_bn_parameters_bias_
        )
        l_self_modules_level3_modules_tree1_modules_project_modules_0_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_project_modules_0_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_project_modules_1_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_project_modules_1_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_project_modules_1_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_project_modules_1_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_project_modules_1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_project_modules_1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_project_modules_1_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_project_modules_1_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_root_modules_conv_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_root_modules_conv_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_root_modules_bn_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_root_modules_bn_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_root_modules_bn_parameters_bias_ = (
            L_self_modules_level3_modules_tree1_modules_root_modules_bn_parameters_bias_
        )
        l_self_modules_level3_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_root_modules_conv_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_root_modules_conv_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_root_modules_bn_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_root_modules_bn_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_root_modules_bn_parameters_bias_ = (
            L_self_modules_level3_modules_tree2_modules_root_modules_bn_parameters_bias_
        )
        l_self_modules_level4_modules_tree1_modules_project_modules_0_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_project_modules_0_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_project_modules_1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_project_modules_1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_project_modules_1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_project_modules_1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_project_modules_1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_project_modules_1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_project_modules_1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_project_modules_1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_root_modules_conv_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_root_modules_conv_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_root_modules_bn_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_root_modules_bn_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_root_modules_bn_parameters_bias_ = (
            L_self_modules_level4_modules_tree1_modules_root_modules_bn_parameters_bias_
        )
        l_self_modules_level4_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_root_modules_conv_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_root_modules_conv_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_root_modules_bn_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_root_modules_bn_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_root_modules_bn_parameters_bias_ = (
            L_self_modules_level4_modules_tree2_modules_root_modules_bn_parameters_bias_
        )
        l_self_modules_level5_modules_project_modules_0_parameters_weight_ = (
            L_self_modules_level5_modules_project_modules_0_parameters_weight_
        )
        l_self_modules_level5_modules_project_modules_1_buffers_running_mean_ = (
            L_self_modules_level5_modules_project_modules_1_buffers_running_mean_
        )
        l_self_modules_level5_modules_project_modules_1_buffers_running_var_ = (
            L_self_modules_level5_modules_project_modules_1_buffers_running_var_
        )
        l_self_modules_level5_modules_project_modules_1_parameters_weight_ = (
            L_self_modules_level5_modules_project_modules_1_parameters_weight_
        )
        l_self_modules_level5_modules_project_modules_1_parameters_bias_ = (
            L_self_modules_level5_modules_project_modules_1_parameters_bias_
        )
        l_self_modules_level5_modules_tree1_modules_conv1_parameters_weight_ = (
            L_self_modules_level5_modules_tree1_modules_conv1_parameters_weight_
        )
        l_self_modules_level5_modules_tree1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_level5_modules_tree1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_level5_modules_tree1_modules_bn1_buffers_running_var_ = (
            L_self_modules_level5_modules_tree1_modules_bn1_buffers_running_var_
        )
        l_self_modules_level5_modules_tree1_modules_bn1_parameters_weight_ = (
            L_self_modules_level5_modules_tree1_modules_bn1_parameters_weight_
        )
        l_self_modules_level5_modules_tree1_modules_bn1_parameters_bias_ = (
            L_self_modules_level5_modules_tree1_modules_bn1_parameters_bias_
        )
        l_self_modules_level5_modules_tree1_modules_conv2_parameters_weight_ = (
            L_self_modules_level5_modules_tree1_modules_conv2_parameters_weight_
        )
        l_self_modules_level5_modules_tree1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_level5_modules_tree1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_level5_modules_tree1_modules_bn2_buffers_running_var_ = (
            L_self_modules_level5_modules_tree1_modules_bn2_buffers_running_var_
        )
        l_self_modules_level5_modules_tree1_modules_bn2_parameters_weight_ = (
            L_self_modules_level5_modules_tree1_modules_bn2_parameters_weight_
        )
        l_self_modules_level5_modules_tree1_modules_bn2_parameters_bias_ = (
            L_self_modules_level5_modules_tree1_modules_bn2_parameters_bias_
        )
        l_self_modules_level5_modules_tree2_modules_conv1_parameters_weight_ = (
            L_self_modules_level5_modules_tree2_modules_conv1_parameters_weight_
        )
        l_self_modules_level5_modules_tree2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_level5_modules_tree2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_level5_modules_tree2_modules_bn1_buffers_running_var_ = (
            L_self_modules_level5_modules_tree2_modules_bn1_buffers_running_var_
        )
        l_self_modules_level5_modules_tree2_modules_bn1_parameters_weight_ = (
            L_self_modules_level5_modules_tree2_modules_bn1_parameters_weight_
        )
        l_self_modules_level5_modules_tree2_modules_bn1_parameters_bias_ = (
            L_self_modules_level5_modules_tree2_modules_bn1_parameters_bias_
        )
        l_self_modules_level5_modules_tree2_modules_conv2_parameters_weight_ = (
            L_self_modules_level5_modules_tree2_modules_conv2_parameters_weight_
        )
        l_self_modules_level5_modules_tree2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_level5_modules_tree2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_level5_modules_tree2_modules_bn2_buffers_running_var_ = (
            L_self_modules_level5_modules_tree2_modules_bn2_buffers_running_var_
        )
        l_self_modules_level5_modules_tree2_modules_bn2_parameters_weight_ = (
            L_self_modules_level5_modules_tree2_modules_bn2_parameters_weight_
        )
        l_self_modules_level5_modules_tree2_modules_bn2_parameters_bias_ = (
            L_self_modules_level5_modules_tree2_modules_bn2_parameters_bias_
        )
        l_self_modules_level5_modules_root_modules_conv_parameters_weight_ = (
            L_self_modules_level5_modules_root_modules_conv_parameters_weight_
        )
        l_self_modules_level5_modules_root_modules_bn_buffers_running_mean_ = (
            L_self_modules_level5_modules_root_modules_bn_buffers_running_mean_
        )
        l_self_modules_level5_modules_root_modules_bn_buffers_running_var_ = (
            L_self_modules_level5_modules_root_modules_bn_buffers_running_var_
        )
        l_self_modules_level5_modules_root_modules_bn_parameters_weight_ = (
            L_self_modules_level5_modules_root_modules_bn_parameters_weight_
        )
        l_self_modules_level5_modules_root_modules_bn_parameters_bias_ = (
            L_self_modules_level5_modules_root_modules_bn_parameters_bias_
        )
        l_self_modules_fc_parameters_weight_ = L_self_modules_fc_parameters_weight_
        l_self_modules_fc_parameters_bias_ = L_self_modules_fc_parameters_bias_
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_base_layer_modules_0_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_base_layer_modules_0_parameters_weight_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_base_layer_modules_1_buffers_running_mean_,
            l_self_modules_base_layer_modules_1_buffers_running_var_,
            l_self_modules_base_layer_modules_1_parameters_weight_,
            l_self_modules_base_layer_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_base_layer_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_base_layer_modules_1_buffers_running_var_
        ) = (
            l_self_modules_base_layer_modules_1_parameters_weight_
        ) = l_self_modules_base_layer_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_level0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_level0_modules_0_parameters_weight_ = None
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_level0_modules_1_buffers_running_mean_,
            l_self_modules_level0_modules_1_buffers_running_var_,
            l_self_modules_level0_modules_1_parameters_weight_,
            l_self_modules_level0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = (
            l_self_modules_level0_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_level0_modules_1_buffers_running_var_
        ) = (
            l_self_modules_level0_modules_1_parameters_weight_
        ) = l_self_modules_level0_modules_1_parameters_bias_ = None
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_level1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_level1_modules_0_parameters_weight_ = None
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_level1_modules_1_buffers_running_mean_,
            l_self_modules_level1_modules_1_buffers_running_var_,
            l_self_modules_level1_modules_1_parameters_weight_,
            l_self_modules_level1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = (
            l_self_modules_level1_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_level1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_level1_modules_1_parameters_weight_
        ) = l_self_modules_level1_modules_1_parameters_bias_ = None
        input_9 = torch.nn.functional.relu(input_8, inplace=True)
        input_8 = None
        bottom = torch.nn.functional.max_pool2d(
            input_9, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_10 = torch.conv2d(
            bottom,
            l_self_modules_level2_modules_project_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        bottom = (
            l_self_modules_level2_modules_project_modules_0_parameters_weight_
        ) = None
        input_11 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_level2_modules_project_modules_1_buffers_running_mean_,
            l_self_modules_level2_modules_project_modules_1_buffers_running_var_,
            l_self_modules_level2_modules_project_modules_1_parameters_weight_,
            l_self_modules_level2_modules_project_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_10 = (
            l_self_modules_level2_modules_project_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_level2_modules_project_modules_1_buffers_running_var_
        ) = (
            l_self_modules_level2_modules_project_modules_1_parameters_weight_
        ) = l_self_modules_level2_modules_project_modules_1_parameters_bias_ = None
        out = torch.conv2d(
            input_9,
            l_self_modules_level2_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_9 = (
            l_self_modules_level2_modules_tree1_modules_conv1_parameters_weight_
        ) = None
        out_1 = torch.nn.functional.batch_norm(
            out,
            l_self_modules_level2_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level2_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level2_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level2_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out = (
            l_self_modules_level2_modules_tree1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_level2_modules_tree1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_level2_modules_tree1_modules_bn1_parameters_weight_
        ) = l_self_modules_level2_modules_tree1_modules_bn1_parameters_bias_ = None
        out_2 = torch.nn.functional.relu(out_1, inplace=True)
        out_1 = None
        out_3 = torch.conv2d(
            out_2,
            l_self_modules_level2_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_2 = (
            l_self_modules_level2_modules_tree1_modules_conv2_parameters_weight_
        ) = None
        out_4 = torch.nn.functional.batch_norm(
            out_3,
            l_self_modules_level2_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level2_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level2_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level2_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_3 = (
            l_self_modules_level2_modules_tree1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_level2_modules_tree1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_level2_modules_tree1_modules_bn2_parameters_weight_
        ) = l_self_modules_level2_modules_tree1_modules_bn2_parameters_bias_ = None
        out_4 += input_11
        out_5 = out_4
        out_4 = input_11 = None
        out_6 = torch.nn.functional.relu(out_5, inplace=True)
        out_5 = None
        out_7 = torch.conv2d(
            out_6,
            l_self_modules_level2_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_level2_modules_tree2_modules_conv1_parameters_weight_ = None
        out_8 = torch.nn.functional.batch_norm(
            out_7,
            l_self_modules_level2_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level2_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level2_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level2_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_7 = (
            l_self_modules_level2_modules_tree2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_level2_modules_tree2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_level2_modules_tree2_modules_bn1_parameters_weight_
        ) = l_self_modules_level2_modules_tree2_modules_bn1_parameters_bias_ = None
        out_9 = torch.nn.functional.relu(out_8, inplace=True)
        out_8 = None
        out_10 = torch.conv2d(
            out_9,
            l_self_modules_level2_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_9 = (
            l_self_modules_level2_modules_tree2_modules_conv2_parameters_weight_
        ) = None
        out_11 = torch.nn.functional.batch_norm(
            out_10,
            l_self_modules_level2_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level2_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level2_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level2_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_10 = (
            l_self_modules_level2_modules_tree2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_level2_modules_tree2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_level2_modules_tree2_modules_bn2_parameters_weight_
        ) = l_self_modules_level2_modules_tree2_modules_bn2_parameters_bias_ = None
        out_11 += out_6
        out_12 = out_11
        out_11 = None
        out_13 = torch.nn.functional.relu(out_12, inplace=True)
        out_12 = None
        cat = torch.cat([out_13, out_6], 1)
        out_13 = out_6 = None
        x = torch.conv2d(
            cat,
            l_self_modules_level2_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = l_self_modules_level2_modules_root_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_level2_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level2_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level2_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level2_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_level2_modules_root_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_level2_modules_root_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_level2_modules_root_modules_bn_parameters_weight_
        ) = l_self_modules_level2_modules_root_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        bottom_1 = torch.nn.functional.max_pool2d(
            x_2, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        bottom_2 = torch.nn.functional.max_pool2d(
            x_2, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_12 = torch.conv2d(
            bottom_2,
            l_self_modules_level3_modules_tree1_modules_project_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        bottom_2 = l_self_modules_level3_modules_tree1_modules_project_modules_0_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_level3_modules_tree1_modules_project_modules_1_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_project_modules_1_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_project_modules_1_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_project_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_level3_modules_tree1_modules_project_modules_1_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_project_modules_1_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_project_modules_1_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_project_modules_1_parameters_bias_ = (None)
        out_14 = torch.conv2d(
            x_2,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_level3_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = (None)
        out_15 = torch.nn.functional.batch_norm(
            out_14,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_14 = l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_16 = torch.nn.functional.relu(out_15, inplace=True)
        out_15 = None
        out_17 = torch.conv2d(
            out_16,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_16 = l_self_modules_level3_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_18 = torch.nn.functional.batch_norm(
            out_17,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_17 = l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_18 += input_13
        out_19 = out_18
        out_18 = input_13 = None
        out_20 = torch.nn.functional.relu(out_19, inplace=True)
        out_19 = None
        out_21 = torch.conv2d(
            out_20,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_level3_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_22 = torch.nn.functional.batch_norm(
            out_21,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_21 = l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_23 = torch.nn.functional.relu(out_22, inplace=True)
        out_22 = None
        out_24 = torch.conv2d(
            out_23,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_23 = l_self_modules_level3_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_25 = torch.nn.functional.batch_norm(
            out_24,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_24 = l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_25 += out_20
        out_26 = out_25
        out_25 = None
        out_27 = torch.nn.functional.relu(out_26, inplace=True)
        out_26 = None
        cat_1 = torch.cat([out_27, out_20], 1)
        out_27 = out_20 = None
        x_3 = torch.conv2d(
            cat_1,
            l_self_modules_level3_modules_tree1_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = l_self_modules_level3_modules_tree1_modules_root_modules_conv_parameters_weight_ = (None)
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_level3_modules_tree1_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_level3_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_root_modules_bn_parameters_weight_ = (
            l_self_modules_level3_modules_tree1_modules_root_modules_bn_parameters_bias_
        ) = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        out_28 = torch.conv2d(
            x_5,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_level3_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_29 = torch.nn.functional.batch_norm(
            out_28,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_28 = l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_30 = torch.nn.functional.relu(out_29, inplace=True)
        out_29 = None
        out_31 = torch.conv2d(
            out_30,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_30 = l_self_modules_level3_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_32 = torch.nn.functional.batch_norm(
            out_31,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_31 = l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_32 += x_5
        out_33 = out_32
        out_32 = None
        out_34 = torch.nn.functional.relu(out_33, inplace=True)
        out_33 = None
        out_35 = torch.conv2d(
            out_34,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_level3_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_36 = torch.nn.functional.batch_norm(
            out_35,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_35 = l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_37 = torch.nn.functional.relu(out_36, inplace=True)
        out_36 = None
        out_38 = torch.conv2d(
            out_37,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_37 = l_self_modules_level3_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_39 = torch.nn.functional.batch_norm(
            out_38,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_38 = l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_39 += out_34
        out_40 = out_39
        out_39 = None
        out_41 = torch.nn.functional.relu(out_40, inplace=True)
        out_40 = None
        cat_2 = torch.cat([out_41, out_34, bottom_1, x_5], 1)
        out_41 = out_34 = bottom_1 = x_5 = None
        x_6 = torch.conv2d(
            cat_2,
            l_self_modules_level3_modules_tree2_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = l_self_modules_level3_modules_tree2_modules_root_modules_conv_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_level3_modules_tree2_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_level3_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_root_modules_bn_parameters_weight_ = (
            l_self_modules_level3_modules_tree2_modules_root_modules_bn_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        bottom_3 = torch.nn.functional.max_pool2d(
            x_8, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        bottom_4 = torch.nn.functional.max_pool2d(
            x_8, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_14 = torch.conv2d(
            bottom_4,
            l_self_modules_level4_modules_tree1_modules_project_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        bottom_4 = l_self_modules_level4_modules_tree1_modules_project_modules_0_parameters_weight_ = (None)
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_level4_modules_tree1_modules_project_modules_1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_project_modules_1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_project_modules_1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_project_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_level4_modules_tree1_modules_project_modules_1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_project_modules_1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_project_modules_1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_project_modules_1_parameters_bias_ = (None)
        out_42 = torch.conv2d(
            x_8,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_level4_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = (None)
        out_43 = torch.nn.functional.batch_norm(
            out_42,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_42 = l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_44 = torch.nn.functional.relu(out_43, inplace=True)
        out_43 = None
        out_45 = torch.conv2d(
            out_44,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_44 = l_self_modules_level4_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_46 = torch.nn.functional.batch_norm(
            out_45,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_45 = l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_46 += input_15
        out_47 = out_46
        out_46 = input_15 = None
        out_48 = torch.nn.functional.relu(out_47, inplace=True)
        out_47 = None
        out_49 = torch.conv2d(
            out_48,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_50 = torch.nn.functional.batch_norm(
            out_49,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_49 = l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_51 = torch.nn.functional.relu(out_50, inplace=True)
        out_50 = None
        out_52 = torch.conv2d(
            out_51,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_51 = l_self_modules_level4_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_53 = torch.nn.functional.batch_norm(
            out_52,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_52 = l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_53 += out_48
        out_54 = out_53
        out_53 = None
        out_55 = torch.nn.functional.relu(out_54, inplace=True)
        out_54 = None
        cat_3 = torch.cat([out_55, out_48], 1)
        out_55 = out_48 = None
        x_9 = torch.conv2d(
            cat_3,
            l_self_modules_level4_modules_tree1_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = l_self_modules_level4_modules_tree1_modules_root_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_level4_modules_tree1_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_level4_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_root_modules_bn_parameters_weight_ = (
            l_self_modules_level4_modules_tree1_modules_root_modules_bn_parameters_bias_
        ) = None
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        out_56 = torch.conv2d(
            x_11,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_57 = torch.nn.functional.batch_norm(
            out_56,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_56 = l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_58 = torch.nn.functional.relu(out_57, inplace=True)
        out_57 = None
        out_59 = torch.conv2d(
            out_58,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_58 = l_self_modules_level4_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_60 = torch.nn.functional.batch_norm(
            out_59,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_59 = l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_60 += x_11
        out_61 = out_60
        out_60 = None
        out_62 = torch.nn.functional.relu(out_61, inplace=True)
        out_61 = None
        out_63 = torch.conv2d(
            out_62,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_64 = torch.nn.functional.batch_norm(
            out_63,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_63 = l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_65 = torch.nn.functional.relu(out_64, inplace=True)
        out_64 = None
        out_66 = torch.conv2d(
            out_65,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_65 = l_self_modules_level4_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_67 = torch.nn.functional.batch_norm(
            out_66,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_66 = l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_67 += out_62
        out_68 = out_67
        out_67 = None
        out_69 = torch.nn.functional.relu(out_68, inplace=True)
        out_68 = None
        cat_4 = torch.cat([out_69, out_62, bottom_3, x_11], 1)
        out_69 = out_62 = bottom_3 = x_11 = None
        x_12 = torch.conv2d(
            cat_4,
            l_self_modules_level4_modules_tree2_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_4 = l_self_modules_level4_modules_tree2_modules_root_modules_conv_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_level4_modules_tree2_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_level4_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_root_modules_bn_parameters_weight_ = (
            l_self_modules_level4_modules_tree2_modules_root_modules_bn_parameters_bias_
        ) = None
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        bottom_5 = torch.nn.functional.max_pool2d(
            x_14, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_16 = torch.conv2d(
            bottom_5,
            l_self_modules_level5_modules_project_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level5_modules_project_modules_0_parameters_weight_ = None
        input_17 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_level5_modules_project_modules_1_buffers_running_mean_,
            l_self_modules_level5_modules_project_modules_1_buffers_running_var_,
            l_self_modules_level5_modules_project_modules_1_parameters_weight_,
            l_self_modules_level5_modules_project_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_16 = (
            l_self_modules_level5_modules_project_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_project_modules_1_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_project_modules_1_parameters_weight_
        ) = l_self_modules_level5_modules_project_modules_1_parameters_bias_ = None
        out_70 = torch.conv2d(
            x_14,
            l_self_modules_level5_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_14 = (
            l_self_modules_level5_modules_tree1_modules_conv1_parameters_weight_
        ) = None
        out_71 = torch.nn.functional.batch_norm(
            out_70,
            l_self_modules_level5_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level5_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level5_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level5_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_70 = (
            l_self_modules_level5_modules_tree1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_tree1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_tree1_modules_bn1_parameters_weight_
        ) = l_self_modules_level5_modules_tree1_modules_bn1_parameters_bias_ = None
        out_72 = torch.nn.functional.relu(out_71, inplace=True)
        out_71 = None
        out_73 = torch.conv2d(
            out_72,
            l_self_modules_level5_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_72 = (
            l_self_modules_level5_modules_tree1_modules_conv2_parameters_weight_
        ) = None
        out_74 = torch.nn.functional.batch_norm(
            out_73,
            l_self_modules_level5_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level5_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level5_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level5_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_73 = (
            l_self_modules_level5_modules_tree1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_tree1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_tree1_modules_bn2_parameters_weight_
        ) = l_self_modules_level5_modules_tree1_modules_bn2_parameters_bias_ = None
        out_74 += input_17
        out_75 = out_74
        out_74 = input_17 = None
        out_76 = torch.nn.functional.relu(out_75, inplace=True)
        out_75 = None
        out_77 = torch.conv2d(
            out_76,
            l_self_modules_level5_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_level5_modules_tree2_modules_conv1_parameters_weight_ = None
        out_78 = torch.nn.functional.batch_norm(
            out_77,
            l_self_modules_level5_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level5_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level5_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level5_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_77 = (
            l_self_modules_level5_modules_tree2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_tree2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_tree2_modules_bn1_parameters_weight_
        ) = l_self_modules_level5_modules_tree2_modules_bn1_parameters_bias_ = None
        out_79 = torch.nn.functional.relu(out_78, inplace=True)
        out_78 = None
        out_80 = torch.conv2d(
            out_79,
            l_self_modules_level5_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_79 = (
            l_self_modules_level5_modules_tree2_modules_conv2_parameters_weight_
        ) = None
        out_81 = torch.nn.functional.batch_norm(
            out_80,
            l_self_modules_level5_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level5_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level5_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level5_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_80 = (
            l_self_modules_level5_modules_tree2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_tree2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_tree2_modules_bn2_parameters_weight_
        ) = l_self_modules_level5_modules_tree2_modules_bn2_parameters_bias_ = None
        out_81 += out_76
        out_82 = out_81
        out_81 = None
        out_83 = torch.nn.functional.relu(out_82, inplace=True)
        out_82 = None
        cat_5 = torch.cat([out_83, out_76, bottom_5], 1)
        out_83 = out_76 = bottom_5 = None
        x_15 = torch.conv2d(
            cat_5,
            l_self_modules_level5_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_5 = (
            l_self_modules_level5_modules_root_modules_conv_parameters_weight_
        ) = None
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_level5_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level5_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level5_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level5_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = (
            l_self_modules_level5_modules_root_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_root_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_root_modules_bn_parameters_weight_
        ) = l_self_modules_level5_modules_root_modules_bn_parameters_bias_ = None
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.nn.functional.adaptive_avg_pool2d(x_17, 1)
        x_17 = None
        x_19 = torch.nn.functional.dropout(x_18, 0.0, False, False)
        x_18 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        x_21 = x_20.flatten(1, -1)
        x_20 = None
        return (x_21,)
