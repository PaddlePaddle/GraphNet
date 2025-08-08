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
        L_self_modules_level2_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level2_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level2_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_level2_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level2_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level2_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level2_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level2_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level2_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_level5_modules_tree1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level5_modules_tree1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level5_modules_tree1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_level5_modules_tree2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_level5_modules_tree2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_level5_modules_tree2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_level5_modules_tree2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_level2_modules_tree1_modules_conv3_parameters_weight_ = (
            L_self_modules_level2_modules_tree1_modules_conv3_parameters_weight_
        )
        l_self_modules_level2_modules_tree1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_level2_modules_tree1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_level2_modules_tree1_modules_bn3_buffers_running_var_ = (
            L_self_modules_level2_modules_tree1_modules_bn3_buffers_running_var_
        )
        l_self_modules_level2_modules_tree1_modules_bn3_parameters_weight_ = (
            L_self_modules_level2_modules_tree1_modules_bn3_parameters_weight_
        )
        l_self_modules_level2_modules_tree1_modules_bn3_parameters_bias_ = (
            L_self_modules_level2_modules_tree1_modules_bn3_parameters_bias_
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
        l_self_modules_level2_modules_tree2_modules_conv3_parameters_weight_ = (
            L_self_modules_level2_modules_tree2_modules_conv3_parameters_weight_
        )
        l_self_modules_level2_modules_tree2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_level2_modules_tree2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_level2_modules_tree2_modules_bn3_buffers_running_var_ = (
            L_self_modules_level2_modules_tree2_modules_bn3_buffers_running_var_
        )
        l_self_modules_level2_modules_tree2_modules_bn3_parameters_weight_ = (
            L_self_modules_level2_modules_tree2_modules_bn3_parameters_weight_
        )
        l_self_modules_level2_modules_tree2_modules_bn3_parameters_bias_ = (
            L_self_modules_level2_modules_tree2_modules_bn3_parameters_bias_
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
        l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_0_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_0_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_
        l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_ = L_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_
        l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_ = L_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_0_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_0_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_ = L_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_ = L_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_
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
        l_self_modules_level5_modules_tree1_modules_conv3_parameters_weight_ = (
            L_self_modules_level5_modules_tree1_modules_conv3_parameters_weight_
        )
        l_self_modules_level5_modules_tree1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_level5_modules_tree1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_level5_modules_tree1_modules_bn3_buffers_running_var_ = (
            L_self_modules_level5_modules_tree1_modules_bn3_buffers_running_var_
        )
        l_self_modules_level5_modules_tree1_modules_bn3_parameters_weight_ = (
            L_self_modules_level5_modules_tree1_modules_bn3_parameters_weight_
        )
        l_self_modules_level5_modules_tree1_modules_bn3_parameters_bias_ = (
            L_self_modules_level5_modules_tree1_modules_bn3_parameters_bias_
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
        l_self_modules_level5_modules_tree2_modules_conv3_parameters_weight_ = (
            L_self_modules_level5_modules_tree2_modules_conv3_parameters_weight_
        )
        l_self_modules_level5_modules_tree2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_level5_modules_tree2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_level5_modules_tree2_modules_bn3_buffers_running_var_ = (
            L_self_modules_level5_modules_tree2_modules_bn3_buffers_running_var_
        )
        l_self_modules_level5_modules_tree2_modules_bn3_parameters_weight_ = (
            L_self_modules_level5_modules_tree2_modules_bn3_parameters_weight_
        )
        l_self_modules_level5_modules_tree2_modules_bn3_parameters_bias_ = (
            L_self_modules_level5_modules_tree2_modules_bn3_parameters_bias_
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
            (1, 1),
            (0, 0),
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
            (2, 2),
            (1, 1),
            (1, 1),
            32,
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
        out_5 = torch.nn.functional.relu(out_4, inplace=True)
        out_4 = None
        out_6 = torch.conv2d(
            out_5,
            l_self_modules_level2_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_5 = (
            l_self_modules_level2_modules_tree1_modules_conv3_parameters_weight_
        ) = None
        out_7 = torch.nn.functional.batch_norm(
            out_6,
            l_self_modules_level2_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level2_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level2_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level2_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_6 = (
            l_self_modules_level2_modules_tree1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_level2_modules_tree1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_level2_modules_tree1_modules_bn3_parameters_weight_
        ) = l_self_modules_level2_modules_tree1_modules_bn3_parameters_bias_ = None
        out_7 += input_11
        out_8 = out_7
        out_7 = input_11 = None
        out_9 = torch.nn.functional.relu(out_8, inplace=True)
        out_8 = None
        out_10 = torch.conv2d(
            out_9,
            l_self_modules_level2_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level2_modules_tree2_modules_conv1_parameters_weight_ = None
        out_11 = torch.nn.functional.batch_norm(
            out_10,
            l_self_modules_level2_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level2_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level2_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level2_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_10 = (
            l_self_modules_level2_modules_tree2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_level2_modules_tree2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_level2_modules_tree2_modules_bn1_parameters_weight_
        ) = l_self_modules_level2_modules_tree2_modules_bn1_parameters_bias_ = None
        out_12 = torch.nn.functional.relu(out_11, inplace=True)
        out_11 = None
        out_13 = torch.conv2d(
            out_12,
            l_self_modules_level2_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_12 = (
            l_self_modules_level2_modules_tree2_modules_conv2_parameters_weight_
        ) = None
        out_14 = torch.nn.functional.batch_norm(
            out_13,
            l_self_modules_level2_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level2_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level2_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level2_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_13 = (
            l_self_modules_level2_modules_tree2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_level2_modules_tree2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_level2_modules_tree2_modules_bn2_parameters_weight_
        ) = l_self_modules_level2_modules_tree2_modules_bn2_parameters_bias_ = None
        out_15 = torch.nn.functional.relu(out_14, inplace=True)
        out_14 = None
        out_16 = torch.conv2d(
            out_15,
            l_self_modules_level2_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_15 = (
            l_self_modules_level2_modules_tree2_modules_conv3_parameters_weight_
        ) = None
        out_17 = torch.nn.functional.batch_norm(
            out_16,
            l_self_modules_level2_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level2_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level2_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level2_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_16 = (
            l_self_modules_level2_modules_tree2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_level2_modules_tree2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_level2_modules_tree2_modules_bn3_parameters_weight_
        ) = l_self_modules_level2_modules_tree2_modules_bn3_parameters_bias_ = None
        out_17 += out_9
        out_18 = out_17
        out_17 = None
        out_19 = torch.nn.functional.relu(out_18, inplace=True)
        out_18 = None
        cat = torch.cat([out_19, out_9], 1)
        out_9 = None
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
        x_1 += out_19
        x_2 = x_1
        x_1 = out_19 = None
        x_3 = torch.nn.functional.relu(x_2, inplace=True)
        x_2 = None
        bottom_1 = torch.nn.functional.max_pool2d(
            x_3, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        bottom_2 = torch.nn.functional.max_pool2d(
            x_3, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        bottom_2 = None
        bottom_3 = torch.nn.functional.max_pool2d(
            x_3, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_12 = torch.conv2d(
            bottom_3,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        bottom_3 = l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_0_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_project_modules_1_parameters_bias_ = (None)
        out_20 = torch.conv2d(
            x_3,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = (None)
        out_21 = torch.nn.functional.batch_norm(
            out_20,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_20 = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_22 = torch.nn.functional.relu(out_21, inplace=True)
        out_21 = None
        out_23 = torch.conv2d(
            out_22,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        out_22 = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_24 = torch.nn.functional.batch_norm(
            out_23,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_23 = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_25 = torch.nn.functional.relu(out_24, inplace=True)
        out_24 = None
        out_26 = torch.conv2d(
            out_25,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_25 = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_27 = torch.nn.functional.batch_norm(
            out_26,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_26 = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_27 += input_13
        out_28 = out_27
        out_27 = input_13 = None
        out_29 = torch.nn.functional.relu(out_28, inplace=True)
        out_28 = None
        out_30 = torch.conv2d(
            out_29,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_31 = torch.nn.functional.batch_norm(
            out_30,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_30 = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_32 = torch.nn.functional.relu(out_31, inplace=True)
        out_31 = None
        out_33 = torch.conv2d(
            out_32,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_32 = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_34 = torch.nn.functional.batch_norm(
            out_33,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_33 = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_35 = torch.nn.functional.relu(out_34, inplace=True)
        out_34 = None
        out_36 = torch.conv2d(
            out_35,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_35 = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_37 = torch.nn.functional.batch_norm(
            out_36,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_36 = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_37 += out_29
        out_38 = out_37
        out_37 = None
        out_39 = torch.nn.functional.relu(out_38, inplace=True)
        out_38 = None
        cat_1 = torch.cat([out_39, out_29], 1)
        out_29 = None
        x_4 = torch.conv2d(
            cat_1,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_ = (None)
        x_5 = torch.nn.functional.batch_norm(
            x_4,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_4 = l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_ = (None)
        x_5 += out_39
        x_6 = x_5
        x_5 = out_39 = None
        x_7 = torch.nn.functional.relu(x_6, inplace=True)
        x_6 = None
        out_40 = torch.conv2d(
            x_7,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_41 = torch.nn.functional.batch_norm(
            out_40,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_40 = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_42 = torch.nn.functional.relu(out_41, inplace=True)
        out_41 = None
        out_43 = torch.conv2d(
            out_42,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_42 = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_44 = torch.nn.functional.batch_norm(
            out_43,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_43 = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_45 = torch.nn.functional.relu(out_44, inplace=True)
        out_44 = None
        out_46 = torch.conv2d(
            out_45,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_45 = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_47 = torch.nn.functional.batch_norm(
            out_46,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_46 = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_47 += x_7
        out_48 = out_47
        out_47 = None
        out_49 = torch.nn.functional.relu(out_48, inplace=True)
        out_48 = None
        out_50 = torch.conv2d(
            out_49,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_51 = torch.nn.functional.batch_norm(
            out_50,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_50 = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_52 = torch.nn.functional.relu(out_51, inplace=True)
        out_51 = None
        out_53 = torch.conv2d(
            out_52,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_52 = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_54 = torch.nn.functional.batch_norm(
            out_53,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_53 = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_55 = torch.nn.functional.relu(out_54, inplace=True)
        out_54 = None
        out_56 = torch.conv2d(
            out_55,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_55 = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_57 = torch.nn.functional.batch_norm(
            out_56,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_56 = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_57 += out_49
        out_58 = out_57
        out_57 = None
        out_59 = torch.nn.functional.relu(out_58, inplace=True)
        out_58 = None
        cat_2 = torch.cat([out_59, out_49, x_7], 1)
        out_49 = x_7 = None
        x_8 = torch.conv2d(
            cat_2,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_ = (None)
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_8 = l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_ = l_self_modules_level3_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_ = (None)
        x_9 += out_59
        x_10 = x_9
        x_9 = out_59 = None
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        out_60 = torch.conv2d(
            x_11,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_61 = torch.nn.functional.batch_norm(
            out_60,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_60 = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_62 = torch.nn.functional.relu(out_61, inplace=True)
        out_61 = None
        out_63 = torch.conv2d(
            out_62,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_62 = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_64 = torch.nn.functional.batch_norm(
            out_63,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_63 = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_65 = torch.nn.functional.relu(out_64, inplace=True)
        out_64 = None
        out_66 = torch.conv2d(
            out_65,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_65 = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_67 = torch.nn.functional.batch_norm(
            out_66,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_66 = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_67 += x_11
        out_68 = out_67
        out_67 = None
        out_69 = torch.nn.functional.relu(out_68, inplace=True)
        out_68 = None
        out_70 = torch.conv2d(
            out_69,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_71 = torch.nn.functional.batch_norm(
            out_70,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_70 = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_72 = torch.nn.functional.relu(out_71, inplace=True)
        out_71 = None
        out_73 = torch.conv2d(
            out_72,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_72 = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_74 = torch.nn.functional.batch_norm(
            out_73,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_73 = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_75 = torch.nn.functional.relu(out_74, inplace=True)
        out_74 = None
        out_76 = torch.conv2d(
            out_75,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_75 = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_77 = torch.nn.functional.batch_norm(
            out_76,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_76 = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_77 += out_69
        out_78 = out_77
        out_77 = None
        out_79 = torch.nn.functional.relu(out_78, inplace=True)
        out_78 = None
        cat_3 = torch.cat([out_79, out_69], 1)
        out_69 = None
        x_12 = torch.conv2d(
            cat_3,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_ = (None)
        x_13 += out_79
        x_14 = x_13
        x_13 = out_79 = None
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        out_80 = torch.conv2d(
            x_15,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_81 = torch.nn.functional.batch_norm(
            out_80,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_80 = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_82 = torch.nn.functional.relu(out_81, inplace=True)
        out_81 = None
        out_83 = torch.conv2d(
            out_82,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_82 = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_84 = torch.nn.functional.batch_norm(
            out_83,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_83 = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_85 = torch.nn.functional.relu(out_84, inplace=True)
        out_84 = None
        out_86 = torch.conv2d(
            out_85,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_85 = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_87 = torch.nn.functional.batch_norm(
            out_86,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_86 = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_87 += x_15
        out_88 = out_87
        out_87 = None
        out_89 = torch.nn.functional.relu(out_88, inplace=True)
        out_88 = None
        out_90 = torch.conv2d(
            out_89,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_91 = torch.nn.functional.batch_norm(
            out_90,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_90 = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_92 = torch.nn.functional.relu(out_91, inplace=True)
        out_91 = None
        out_93 = torch.conv2d(
            out_92,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_92 = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_94 = torch.nn.functional.batch_norm(
            out_93,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_93 = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_95 = torch.nn.functional.relu(out_94, inplace=True)
        out_94 = None
        out_96 = torch.conv2d(
            out_95,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_95 = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_97 = torch.nn.functional.batch_norm(
            out_96,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_96 = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_97 += out_89
        out_98 = out_97
        out_97 = None
        out_99 = torch.nn.functional.relu(out_98, inplace=True)
        out_98 = None
        cat_4 = torch.cat([out_99, out_89, bottom_1, x_11, x_15], 1)
        out_89 = bottom_1 = x_11 = x_15 = None
        x_16 = torch.conv2d(
            cat_4,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_4 = l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_ = l_self_modules_level3_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_ = (None)
        x_17 += out_99
        x_18 = x_17
        x_17 = out_99 = None
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        bottom_4 = torch.nn.functional.max_pool2d(
            x_19, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        bottom_5 = torch.nn.functional.max_pool2d(
            x_19, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        bottom_5 = None
        bottom_6 = torch.nn.functional.max_pool2d(
            x_19, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        bottom_6 = None
        bottom_7 = torch.nn.functional.max_pool2d(
            x_19, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_14 = torch.conv2d(
            bottom_7,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        bottom_7 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_0_parameters_weight_ = (None)
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_project_modules_1_parameters_bias_ = (None)
        out_100 = torch.conv2d(
            x_19,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = (None)
        out_101 = torch.nn.functional.batch_norm(
            out_100,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_100 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_102 = torch.nn.functional.relu(out_101, inplace=True)
        out_101 = None
        out_103 = torch.conv2d(
            out_102,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        out_102 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_104 = torch.nn.functional.batch_norm(
            out_103,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_103 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_105 = torch.nn.functional.relu(out_104, inplace=True)
        out_104 = None
        out_106 = torch.conv2d(
            out_105,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_105 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_107 = torch.nn.functional.batch_norm(
            out_106,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_106 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_107 += input_15
        out_108 = out_107
        out_107 = input_15 = None
        out_109 = torch.nn.functional.relu(out_108, inplace=True)
        out_108 = None
        out_110 = torch.conv2d(
            out_109,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_111 = torch.nn.functional.batch_norm(
            out_110,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_110 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_112 = torch.nn.functional.relu(out_111, inplace=True)
        out_111 = None
        out_113 = torch.conv2d(
            out_112,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_112 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_114 = torch.nn.functional.batch_norm(
            out_113,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_113 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_115 = torch.nn.functional.relu(out_114, inplace=True)
        out_114 = None
        out_116 = torch.conv2d(
            out_115,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_115 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_117 = torch.nn.functional.batch_norm(
            out_116,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_116 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_117 += out_109
        out_118 = out_117
        out_117 = None
        out_119 = torch.nn.functional.relu(out_118, inplace=True)
        out_118 = None
        cat_5 = torch.cat([out_119, out_109], 1)
        out_109 = None
        x_20 = torch.conv2d(
            cat_5,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_5 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_ = (None)
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_ = (None)
        x_21 += out_119
        x_22 = x_21
        x_21 = out_119 = None
        x_23 = torch.nn.functional.relu(x_22, inplace=True)
        x_22 = None
        out_120 = torch.conv2d(
            x_23,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_121 = torch.nn.functional.batch_norm(
            out_120,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_120 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_122 = torch.nn.functional.relu(out_121, inplace=True)
        out_121 = None
        out_123 = torch.conv2d(
            out_122,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_122 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_124 = torch.nn.functional.batch_norm(
            out_123,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_123 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_125 = torch.nn.functional.relu(out_124, inplace=True)
        out_124 = None
        out_126 = torch.conv2d(
            out_125,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_125 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_127 = torch.nn.functional.batch_norm(
            out_126,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_126 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_127 += x_23
        out_128 = out_127
        out_127 = None
        out_129 = torch.nn.functional.relu(out_128, inplace=True)
        out_128 = None
        out_130 = torch.conv2d(
            out_129,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_131 = torch.nn.functional.batch_norm(
            out_130,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_130 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_132 = torch.nn.functional.relu(out_131, inplace=True)
        out_131 = None
        out_133 = torch.conv2d(
            out_132,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_132 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_134 = torch.nn.functional.batch_norm(
            out_133,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_133 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_135 = torch.nn.functional.relu(out_134, inplace=True)
        out_134 = None
        out_136 = torch.conv2d(
            out_135,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_135 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_137 = torch.nn.functional.batch_norm(
            out_136,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_136 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_137 += out_129
        out_138 = out_137
        out_137 = None
        out_139 = torch.nn.functional.relu(out_138, inplace=True)
        out_138 = None
        cat_6 = torch.cat([out_139, out_129, x_23], 1)
        out_129 = x_23 = None
        x_24 = torch.conv2d(
            cat_6,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_6 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_ = (None)
        x_25 += out_139
        x_26 = x_25
        x_25 = out_139 = None
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        out_140 = torch.conv2d(
            x_27,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_141 = torch.nn.functional.batch_norm(
            out_140,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_140 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_142 = torch.nn.functional.relu(out_141, inplace=True)
        out_141 = None
        out_143 = torch.conv2d(
            out_142,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_142 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_144 = torch.nn.functional.batch_norm(
            out_143,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_143 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_145 = torch.nn.functional.relu(out_144, inplace=True)
        out_144 = None
        out_146 = torch.conv2d(
            out_145,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_145 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_147 = torch.nn.functional.batch_norm(
            out_146,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_146 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_147 += x_27
        out_148 = out_147
        out_147 = None
        out_149 = torch.nn.functional.relu(out_148, inplace=True)
        out_148 = None
        out_150 = torch.conv2d(
            out_149,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_151 = torch.nn.functional.batch_norm(
            out_150,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_150 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_152 = torch.nn.functional.relu(out_151, inplace=True)
        out_151 = None
        out_153 = torch.conv2d(
            out_152,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_152 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_154 = torch.nn.functional.batch_norm(
            out_153,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_153 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_155 = torch.nn.functional.relu(out_154, inplace=True)
        out_154 = None
        out_156 = torch.conv2d(
            out_155,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_155 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_157 = torch.nn.functional.batch_norm(
            out_156,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_156 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_157 += out_149
        out_158 = out_157
        out_157 = None
        out_159 = torch.nn.functional.relu(out_158, inplace=True)
        out_158 = None
        cat_7 = torch.cat([out_159, out_149], 1)
        out_149 = None
        x_28 = torch.conv2d(
            cat_7,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_7 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_ = (None)
        x_29 += out_159
        x_30 = x_29
        x_29 = out_159 = None
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        out_160 = torch.conv2d(
            x_31,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_161 = torch.nn.functional.batch_norm(
            out_160,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_160 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_162 = torch.nn.functional.relu(out_161, inplace=True)
        out_161 = None
        out_163 = torch.conv2d(
            out_162,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_162 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_164 = torch.nn.functional.batch_norm(
            out_163,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_163 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_165 = torch.nn.functional.relu(out_164, inplace=True)
        out_164 = None
        out_166 = torch.conv2d(
            out_165,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_165 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_167 = torch.nn.functional.batch_norm(
            out_166,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_166 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_167 += x_31
        out_168 = out_167
        out_167 = None
        out_169 = torch.nn.functional.relu(out_168, inplace=True)
        out_168 = None
        out_170 = torch.conv2d(
            out_169,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_171 = torch.nn.functional.batch_norm(
            out_170,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_170 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_172 = torch.nn.functional.relu(out_171, inplace=True)
        out_171 = None
        out_173 = torch.conv2d(
            out_172,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_172 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_174 = torch.nn.functional.batch_norm(
            out_173,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_173 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_175 = torch.nn.functional.relu(out_174, inplace=True)
        out_174 = None
        out_176 = torch.conv2d(
            out_175,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_175 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_177 = torch.nn.functional.batch_norm(
            out_176,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_176 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_177 += out_169
        out_178 = out_177
        out_177 = None
        out_179 = torch.nn.functional.relu(out_178, inplace=True)
        out_178 = None
        cat_8 = torch.cat([out_179, out_169, x_27, x_31], 1)
        out_169 = x_27 = x_31 = None
        x_32 = torch.conv2d(
            cat_8,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_8 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_ = l_self_modules_level4_modules_tree1_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_ = (None)
        x_33 += out_179
        x_34 = x_33
        x_33 = out_179 = None
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        out_180 = torch.conv2d(
            x_35,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_181 = torch.nn.functional.batch_norm(
            out_180,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_180 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_182 = torch.nn.functional.relu(out_181, inplace=True)
        out_181 = None
        out_183 = torch.conv2d(
            out_182,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_182 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_184 = torch.nn.functional.batch_norm(
            out_183,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_183 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_185 = torch.nn.functional.relu(out_184, inplace=True)
        out_184 = None
        out_186 = torch.conv2d(
            out_185,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_185 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_187 = torch.nn.functional.batch_norm(
            out_186,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_186 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_187 += x_35
        out_188 = out_187
        out_187 = None
        out_189 = torch.nn.functional.relu(out_188, inplace=True)
        out_188 = None
        out_190 = torch.conv2d(
            out_189,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_191 = torch.nn.functional.batch_norm(
            out_190,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_190 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_192 = torch.nn.functional.relu(out_191, inplace=True)
        out_191 = None
        out_193 = torch.conv2d(
            out_192,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_192 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_194 = torch.nn.functional.batch_norm(
            out_193,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_193 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_195 = torch.nn.functional.relu(out_194, inplace=True)
        out_194 = None
        out_196 = torch.conv2d(
            out_195,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_195 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_197 = torch.nn.functional.batch_norm(
            out_196,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_196 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_197 += out_189
        out_198 = out_197
        out_197 = None
        out_199 = torch.nn.functional.relu(out_198, inplace=True)
        out_198 = None
        cat_9 = torch.cat([out_199, out_189], 1)
        out_189 = None
        x_36 = torch.conv2d(
            cat_9,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_9 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree1_modules_root_modules_bn_parameters_bias_ = (None)
        x_37 += out_199
        x_38 = x_37
        x_37 = out_199 = None
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        out_200 = torch.conv2d(
            x_39,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_201 = torch.nn.functional.batch_norm(
            out_200,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_200 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_202 = torch.nn.functional.relu(out_201, inplace=True)
        out_201 = None
        out_203 = torch.conv2d(
            out_202,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_202 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_204 = torch.nn.functional.batch_norm(
            out_203,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_203 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_205 = torch.nn.functional.relu(out_204, inplace=True)
        out_204 = None
        out_206 = torch.conv2d(
            out_205,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_205 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_207 = torch.nn.functional.batch_norm(
            out_206,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_206 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_207 += x_39
        out_208 = out_207
        out_207 = None
        out_209 = torch.nn.functional.relu(out_208, inplace=True)
        out_208 = None
        out_210 = torch.conv2d(
            out_209,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_211 = torch.nn.functional.batch_norm(
            out_210,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_210 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_212 = torch.nn.functional.relu(out_211, inplace=True)
        out_211 = None
        out_213 = torch.conv2d(
            out_212,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_212 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_214 = torch.nn.functional.batch_norm(
            out_213,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_213 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_215 = torch.nn.functional.relu(out_214, inplace=True)
        out_214 = None
        out_216 = torch.conv2d(
            out_215,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_215 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_217 = torch.nn.functional.batch_norm(
            out_216,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_216 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_217 += out_209
        out_218 = out_217
        out_217 = None
        out_219 = torch.nn.functional.relu(out_218, inplace=True)
        out_218 = None
        cat_10 = torch.cat([out_219, out_209, x_39], 1)
        out_209 = x_39 = None
        x_40 = torch.conv2d(
            cat_10,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_10 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree1_modules_tree2_modules_root_modules_bn_parameters_bias_ = (None)
        x_41 += out_219
        x_42 = x_41
        x_41 = out_219 = None
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        out_220 = torch.conv2d(
            x_43,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_221 = torch.nn.functional.batch_norm(
            out_220,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_220 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_222 = torch.nn.functional.relu(out_221, inplace=True)
        out_221 = None
        out_223 = torch.conv2d(
            out_222,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_222 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_224 = torch.nn.functional.batch_norm(
            out_223,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_223 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_225 = torch.nn.functional.relu(out_224, inplace=True)
        out_224 = None
        out_226 = torch.conv2d(
            out_225,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_225 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_227 = torch.nn.functional.batch_norm(
            out_226,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_226 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_227 += x_43
        out_228 = out_227
        out_227 = None
        out_229 = torch.nn.functional.relu(out_228, inplace=True)
        out_228 = None
        out_230 = torch.conv2d(
            out_229,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_231 = torch.nn.functional.batch_norm(
            out_230,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_230 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_232 = torch.nn.functional.relu(out_231, inplace=True)
        out_231 = None
        out_233 = torch.conv2d(
            out_232,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_232 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_234 = torch.nn.functional.batch_norm(
            out_233,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_233 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_235 = torch.nn.functional.relu(out_234, inplace=True)
        out_234 = None
        out_236 = torch.conv2d(
            out_235,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_235 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_237 = torch.nn.functional.batch_norm(
            out_236,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_236 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_237 += out_229
        out_238 = out_237
        out_237 = None
        out_239 = torch.nn.functional.relu(out_238, inplace=True)
        out_238 = None
        cat_11 = torch.cat([out_239, out_229], 1)
        out_229 = None
        x_44 = torch.conv2d(
            cat_11,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_11 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_conv_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree1_modules_root_modules_bn_parameters_bias_ = (None)
        x_45 += out_239
        x_46 = x_45
        x_45 = out_239 = None
        x_47 = torch.nn.functional.relu(x_46, inplace=True)
        x_46 = None
        out_240 = torch.conv2d(
            x_47,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv1_parameters_weight_ = (
            None
        )
        out_241 = torch.nn.functional.batch_norm(
            out_240,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_240 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn1_parameters_bias_ = (None)
        out_242 = torch.nn.functional.relu(out_241, inplace=True)
        out_241 = None
        out_243 = torch.conv2d(
            out_242,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_242 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv2_parameters_weight_ = (None)
        out_244 = torch.nn.functional.batch_norm(
            out_243,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_243 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn2_parameters_bias_ = (None)
        out_245 = torch.nn.functional.relu(out_244, inplace=True)
        out_244 = None
        out_246 = torch.conv2d(
            out_245,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_245 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_conv3_parameters_weight_ = (None)
        out_247 = torch.nn.functional.batch_norm(
            out_246,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_246 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree1_modules_bn3_parameters_bias_ = (None)
        out_247 += x_47
        out_248 = out_247
        out_247 = None
        out_249 = torch.nn.functional.relu(out_248, inplace=True)
        out_248 = None
        out_250 = torch.conv2d(
            out_249,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv1_parameters_weight_ = (
            None
        )
        out_251 = torch.nn.functional.batch_norm(
            out_250,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_250 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn1_parameters_bias_ = (None)
        out_252 = torch.nn.functional.relu(out_251, inplace=True)
        out_251 = None
        out_253 = torch.conv2d(
            out_252,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_252 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv2_parameters_weight_ = (None)
        out_254 = torch.nn.functional.batch_norm(
            out_253,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_253 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn2_parameters_bias_ = (None)
        out_255 = torch.nn.functional.relu(out_254, inplace=True)
        out_254 = None
        out_256 = torch.conv2d(
            out_255,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_255 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_conv3_parameters_weight_ = (None)
        out_257 = torch.nn.functional.batch_norm(
            out_256,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_256 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_tree2_modules_bn3_parameters_bias_ = (None)
        out_257 += out_249
        out_258 = out_257
        out_257 = None
        out_259 = torch.nn.functional.relu(out_258, inplace=True)
        out_258 = None
        cat_12 = torch.cat([out_259, out_249, bottom_4, x_35, x_43, x_47], 1)
        out_249 = bottom_4 = x_35 = x_43 = x_47 = None
        x_48 = torch.conv2d(
            cat_12,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_12 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_conv_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_mean_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_buffers_running_var_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_weight_ = l_self_modules_level4_modules_tree2_modules_tree2_modules_tree2_modules_root_modules_bn_parameters_bias_ = (None)
        x_49 += out_259
        x_50 = x_49
        x_49 = out_259 = None
        x_51 = torch.nn.functional.relu(x_50, inplace=True)
        x_50 = None
        bottom_8 = torch.nn.functional.max_pool2d(
            x_51, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_16 = torch.conv2d(
            bottom_8,
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
        out_260 = torch.conv2d(
            x_51,
            l_self_modules_level5_modules_tree1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_51 = (
            l_self_modules_level5_modules_tree1_modules_conv1_parameters_weight_
        ) = None
        out_261 = torch.nn.functional.batch_norm(
            out_260,
            l_self_modules_level5_modules_tree1_modules_bn1_buffers_running_mean_,
            l_self_modules_level5_modules_tree1_modules_bn1_buffers_running_var_,
            l_self_modules_level5_modules_tree1_modules_bn1_parameters_weight_,
            l_self_modules_level5_modules_tree1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_260 = (
            l_self_modules_level5_modules_tree1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_tree1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_tree1_modules_bn1_parameters_weight_
        ) = l_self_modules_level5_modules_tree1_modules_bn1_parameters_bias_ = None
        out_262 = torch.nn.functional.relu(out_261, inplace=True)
        out_261 = None
        out_263 = torch.conv2d(
            out_262,
            l_self_modules_level5_modules_tree1_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        out_262 = (
            l_self_modules_level5_modules_tree1_modules_conv2_parameters_weight_
        ) = None
        out_264 = torch.nn.functional.batch_norm(
            out_263,
            l_self_modules_level5_modules_tree1_modules_bn2_buffers_running_mean_,
            l_self_modules_level5_modules_tree1_modules_bn2_buffers_running_var_,
            l_self_modules_level5_modules_tree1_modules_bn2_parameters_weight_,
            l_self_modules_level5_modules_tree1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_263 = (
            l_self_modules_level5_modules_tree1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_tree1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_tree1_modules_bn2_parameters_weight_
        ) = l_self_modules_level5_modules_tree1_modules_bn2_parameters_bias_ = None
        out_265 = torch.nn.functional.relu(out_264, inplace=True)
        out_264 = None
        out_266 = torch.conv2d(
            out_265,
            l_self_modules_level5_modules_tree1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_265 = (
            l_self_modules_level5_modules_tree1_modules_conv3_parameters_weight_
        ) = None
        out_267 = torch.nn.functional.batch_norm(
            out_266,
            l_self_modules_level5_modules_tree1_modules_bn3_buffers_running_mean_,
            l_self_modules_level5_modules_tree1_modules_bn3_buffers_running_var_,
            l_self_modules_level5_modules_tree1_modules_bn3_parameters_weight_,
            l_self_modules_level5_modules_tree1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_266 = (
            l_self_modules_level5_modules_tree1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_tree1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_tree1_modules_bn3_parameters_weight_
        ) = l_self_modules_level5_modules_tree1_modules_bn3_parameters_bias_ = None
        out_267 += input_17
        out_268 = out_267
        out_267 = input_17 = None
        out_269 = torch.nn.functional.relu(out_268, inplace=True)
        out_268 = None
        out_270 = torch.conv2d(
            out_269,
            l_self_modules_level5_modules_tree2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_level5_modules_tree2_modules_conv1_parameters_weight_ = None
        out_271 = torch.nn.functional.batch_norm(
            out_270,
            l_self_modules_level5_modules_tree2_modules_bn1_buffers_running_mean_,
            l_self_modules_level5_modules_tree2_modules_bn1_buffers_running_var_,
            l_self_modules_level5_modules_tree2_modules_bn1_parameters_weight_,
            l_self_modules_level5_modules_tree2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_270 = (
            l_self_modules_level5_modules_tree2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_tree2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_tree2_modules_bn1_parameters_weight_
        ) = l_self_modules_level5_modules_tree2_modules_bn1_parameters_bias_ = None
        out_272 = torch.nn.functional.relu(out_271, inplace=True)
        out_271 = None
        out_273 = torch.conv2d(
            out_272,
            l_self_modules_level5_modules_tree2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        out_272 = (
            l_self_modules_level5_modules_tree2_modules_conv2_parameters_weight_
        ) = None
        out_274 = torch.nn.functional.batch_norm(
            out_273,
            l_self_modules_level5_modules_tree2_modules_bn2_buffers_running_mean_,
            l_self_modules_level5_modules_tree2_modules_bn2_buffers_running_var_,
            l_self_modules_level5_modules_tree2_modules_bn2_parameters_weight_,
            l_self_modules_level5_modules_tree2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_273 = (
            l_self_modules_level5_modules_tree2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_tree2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_tree2_modules_bn2_parameters_weight_
        ) = l_self_modules_level5_modules_tree2_modules_bn2_parameters_bias_ = None
        out_275 = torch.nn.functional.relu(out_274, inplace=True)
        out_274 = None
        out_276 = torch.conv2d(
            out_275,
            l_self_modules_level5_modules_tree2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_275 = (
            l_self_modules_level5_modules_tree2_modules_conv3_parameters_weight_
        ) = None
        out_277 = torch.nn.functional.batch_norm(
            out_276,
            l_self_modules_level5_modules_tree2_modules_bn3_buffers_running_mean_,
            l_self_modules_level5_modules_tree2_modules_bn3_buffers_running_var_,
            l_self_modules_level5_modules_tree2_modules_bn3_parameters_weight_,
            l_self_modules_level5_modules_tree2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_276 = (
            l_self_modules_level5_modules_tree2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_tree2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_tree2_modules_bn3_parameters_weight_
        ) = l_self_modules_level5_modules_tree2_modules_bn3_parameters_bias_ = None
        out_277 += out_269
        out_278 = out_277
        out_277 = None
        out_279 = torch.nn.functional.relu(out_278, inplace=True)
        out_278 = None
        cat_13 = torch.cat([out_279, out_269, bottom_8], 1)
        out_269 = bottom_8 = None
        x_52 = torch.conv2d(
            cat_13,
            l_self_modules_level5_modules_root_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_13 = (
            l_self_modules_level5_modules_root_modules_conv_parameters_weight_
        ) = None
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_level5_modules_root_modules_bn_buffers_running_mean_,
            l_self_modules_level5_modules_root_modules_bn_buffers_running_var_,
            l_self_modules_level5_modules_root_modules_bn_parameters_weight_,
            l_self_modules_level5_modules_root_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = (
            l_self_modules_level5_modules_root_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_level5_modules_root_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_level5_modules_root_modules_bn_parameters_weight_
        ) = l_self_modules_level5_modules_root_modules_bn_parameters_bias_ = None
        x_53 += out_279
        x_54 = x_53
        x_53 = out_279 = None
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.nn.functional.adaptive_avg_pool2d(x_55, 1)
        x_55 = None
        x_57 = torch.nn.functional.dropout(x_56, 0.0, False, False)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        x_59 = x_58.flatten(1, -1)
        x_58 = None
        return (x_59,)
