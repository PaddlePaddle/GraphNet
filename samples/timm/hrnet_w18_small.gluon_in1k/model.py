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
        L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transition1_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transition1_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_transition1_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_transition1_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transition1_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transition1_modules_1_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transition1_modules_1_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_transition1_modules_1_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_transition1_modules_1_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transition1_modules_1_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transition2_modules_2_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transition2_modules_2_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_transition2_modules_2_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_transition2_modules_2_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transition2_modules_2_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_transition3_modules_3_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_transition3_modules_3_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_transition3_modules_3_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_downsamp_modules_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_downsamp_modules_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_downsamp_modules_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_downsamp_modules_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_final_layer_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_final_layer_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_final_layer_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_transition1_modules_0_modules_0_parameters_weight_ = (
            L_self_modules_transition1_modules_0_modules_0_parameters_weight_
        )
        l_self_modules_transition1_modules_0_modules_1_buffers_running_mean_ = (
            L_self_modules_transition1_modules_0_modules_1_buffers_running_mean_
        )
        l_self_modules_transition1_modules_0_modules_1_buffers_running_var_ = (
            L_self_modules_transition1_modules_0_modules_1_buffers_running_var_
        )
        l_self_modules_transition1_modules_0_modules_1_parameters_weight_ = (
            L_self_modules_transition1_modules_0_modules_1_parameters_weight_
        )
        l_self_modules_transition1_modules_0_modules_1_parameters_bias_ = (
            L_self_modules_transition1_modules_0_modules_1_parameters_bias_
        )
        l_self_modules_transition1_modules_1_modules_0_modules_0_parameters_weight_ = (
            L_self_modules_transition1_modules_1_modules_0_modules_0_parameters_weight_
        )
        l_self_modules_transition1_modules_1_modules_0_modules_1_buffers_running_mean_ = L_self_modules_transition1_modules_1_modules_0_modules_1_buffers_running_mean_
        l_self_modules_transition1_modules_1_modules_0_modules_1_buffers_running_var_ = L_self_modules_transition1_modules_1_modules_0_modules_1_buffers_running_var_
        l_self_modules_transition1_modules_1_modules_0_modules_1_parameters_weight_ = (
            L_self_modules_transition1_modules_1_modules_0_modules_1_parameters_weight_
        )
        l_self_modules_transition1_modules_1_modules_0_modules_1_parameters_bias_ = (
            L_self_modules_transition1_modules_1_modules_0_modules_1_parameters_bias_
        )
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_transition2_modules_2_modules_0_modules_0_parameters_weight_ = (
            L_self_modules_transition2_modules_2_modules_0_modules_0_parameters_weight_
        )
        l_self_modules_transition2_modules_2_modules_0_modules_1_buffers_running_mean_ = L_self_modules_transition2_modules_2_modules_0_modules_1_buffers_running_mean_
        l_self_modules_transition2_modules_2_modules_0_modules_1_buffers_running_var_ = L_self_modules_transition2_modules_2_modules_0_modules_1_buffers_running_var_
        l_self_modules_transition2_modules_2_modules_0_modules_1_parameters_weight_ = (
            L_self_modules_transition2_modules_2_modules_0_modules_1_parameters_weight_
        )
        l_self_modules_transition2_modules_2_modules_0_modules_1_parameters_bias_ = (
            L_self_modules_transition2_modules_2_modules_0_modules_1_parameters_bias_
        )
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_ = L_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_
        l_self_modules_transition3_modules_3_modules_0_modules_0_parameters_weight_ = (
            L_self_modules_transition3_modules_3_modules_0_modules_0_parameters_weight_
        )
        l_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_mean_ = L_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_mean_
        l_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_var_ = L_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_var_
        l_self_modules_transition3_modules_3_modules_0_modules_1_parameters_weight_ = (
            L_self_modules_transition3_modules_3_modules_0_modules_1_parameters_weight_
        )
        l_self_modules_transition3_modules_3_modules_0_modules_1_parameters_bias_ = (
            L_self_modules_transition3_modules_3_modules_0_modules_1_parameters_bias_
        )
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_
        l_self_modules_incre_modules_modules_0_modules_0_modules_conv1_parameters_weight_ = L_self_modules_incre_modules_modules_0_modules_0_modules_conv1_parameters_weight_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_weight_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_weight_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_bias_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_bias_
        l_self_modules_incre_modules_modules_0_modules_0_modules_conv2_parameters_weight_ = L_self_modules_incre_modules_modules_0_modules_0_modules_conv2_parameters_weight_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_weight_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_weight_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_bias_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_bias_
        l_self_modules_incre_modules_modules_0_modules_0_modules_conv3_parameters_weight_ = L_self_modules_incre_modules_modules_0_modules_0_modules_conv3_parameters_weight_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_weight_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_weight_
        l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_bias_ = L_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_bias_
        l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_incre_modules_modules_1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_incre_modules_modules_1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_weight_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_weight_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_bias_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_bias_
        l_self_modules_incre_modules_modules_1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_incre_modules_modules_1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_weight_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_weight_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_bias_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_bias_
        l_self_modules_incre_modules_modules_1_modules_0_modules_conv3_parameters_weight_ = L_self_modules_incre_modules_modules_1_modules_0_modules_conv3_parameters_weight_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_weight_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_weight_
        l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_bias_ = L_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_bias_
        l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_downsamp_modules_modules_0_modules_0_parameters_weight_ = (
            L_self_modules_downsamp_modules_modules_0_modules_0_parameters_weight_
        )
        l_self_modules_downsamp_modules_modules_0_modules_0_parameters_bias_ = (
            L_self_modules_downsamp_modules_modules_0_modules_0_parameters_bias_
        )
        l_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_mean_ = (
            L_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_mean_
        )
        l_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_var_ = (
            L_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_var_
        )
        l_self_modules_downsamp_modules_modules_0_modules_1_parameters_weight_ = (
            L_self_modules_downsamp_modules_modules_0_modules_1_parameters_weight_
        )
        l_self_modules_downsamp_modules_modules_0_modules_1_parameters_bias_ = (
            L_self_modules_downsamp_modules_modules_0_modules_1_parameters_bias_
        )
        l_self_modules_incre_modules_modules_2_modules_0_modules_conv1_parameters_weight_ = L_self_modules_incre_modules_modules_2_modules_0_modules_conv1_parameters_weight_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_weight_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_weight_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_bias_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_bias_
        l_self_modules_incre_modules_modules_2_modules_0_modules_conv2_parameters_weight_ = L_self_modules_incre_modules_modules_2_modules_0_modules_conv2_parameters_weight_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_weight_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_weight_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_bias_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_bias_
        l_self_modules_incre_modules_modules_2_modules_0_modules_conv3_parameters_weight_ = L_self_modules_incre_modules_modules_2_modules_0_modules_conv3_parameters_weight_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_weight_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_weight_
        l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_bias_ = L_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_bias_
        l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_downsamp_modules_modules_1_modules_0_parameters_weight_ = (
            L_self_modules_downsamp_modules_modules_1_modules_0_parameters_weight_
        )
        l_self_modules_downsamp_modules_modules_1_modules_0_parameters_bias_ = (
            L_self_modules_downsamp_modules_modules_1_modules_0_parameters_bias_
        )
        l_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_mean_ = (
            L_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_mean_
        )
        l_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_var_ = (
            L_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_var_
        )
        l_self_modules_downsamp_modules_modules_1_modules_1_parameters_weight_ = (
            L_self_modules_downsamp_modules_modules_1_modules_1_parameters_weight_
        )
        l_self_modules_downsamp_modules_modules_1_modules_1_parameters_bias_ = (
            L_self_modules_downsamp_modules_modules_1_modules_1_parameters_bias_
        )
        l_self_modules_incre_modules_modules_3_modules_0_modules_conv1_parameters_weight_ = L_self_modules_incre_modules_modules_3_modules_0_modules_conv1_parameters_weight_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_weight_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_weight_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_bias_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_bias_
        l_self_modules_incre_modules_modules_3_modules_0_modules_conv2_parameters_weight_ = L_self_modules_incre_modules_modules_3_modules_0_modules_conv2_parameters_weight_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_weight_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_weight_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_bias_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_bias_
        l_self_modules_incre_modules_modules_3_modules_0_modules_conv3_parameters_weight_ = L_self_modules_incre_modules_modules_3_modules_0_modules_conv3_parameters_weight_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_weight_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_weight_
        l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_bias_ = L_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_bias_
        l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_downsamp_modules_modules_2_modules_0_parameters_weight_ = (
            L_self_modules_downsamp_modules_modules_2_modules_0_parameters_weight_
        )
        l_self_modules_downsamp_modules_modules_2_modules_0_parameters_bias_ = (
            L_self_modules_downsamp_modules_modules_2_modules_0_parameters_bias_
        )
        l_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_mean_ = (
            L_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_mean_
        )
        l_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_var_ = (
            L_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_var_
        )
        l_self_modules_downsamp_modules_modules_2_modules_1_parameters_weight_ = (
            L_self_modules_downsamp_modules_modules_2_modules_1_parameters_weight_
        )
        l_self_modules_downsamp_modules_modules_2_modules_1_parameters_bias_ = (
            L_self_modules_downsamp_modules_modules_2_modules_1_parameters_bias_
        )
        l_self_modules_final_layer_modules_0_parameters_weight_ = (
            L_self_modules_final_layer_modules_0_parameters_weight_
        )
        l_self_modules_final_layer_modules_0_parameters_bias_ = (
            L_self_modules_final_layer_modules_0_parameters_bias_
        )
        l_self_modules_final_layer_modules_1_buffers_running_mean_ = (
            L_self_modules_final_layer_modules_1_buffers_running_mean_
        )
        l_self_modules_final_layer_modules_1_buffers_running_var_ = (
            L_self_modules_final_layer_modules_1_buffers_running_var_
        )
        l_self_modules_final_layer_modules_1_parameters_weight_ = (
            L_self_modules_final_layer_modules_1_parameters_weight_
        )
        l_self_modules_final_layer_modules_1_parameters_bias_ = (
            L_self_modules_final_layer_modules_1_parameters_bias_
        )
        l_self_modules_classifier_parameters_weight_ = (
            L_self_modules_classifier_parameters_weight_
        )
        l_self_modules_classifier_parameters_bias_ = (
            L_self_modules_classifier_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
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
            (2, 2),
            (1, 1),
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
            l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_ = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_ = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_ = None
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = (
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_ = None
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_ = None
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_ = None
        input_1 = torch.conv2d(
            x_5,
            l_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_13 += input_2
        x_14 = x_13
        x_13 = input_2 = None
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        input_3 = torch.conv2d(
            x_15,
            l_self_modules_transition1_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_transition1_modules_0_modules_0_parameters_weight_ = None
        input_4 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_transition1_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_transition1_modules_0_modules_1_buffers_running_var_,
            l_self_modules_transition1_modules_0_modules_1_parameters_weight_,
            l_self_modules_transition1_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_3 = (
            l_self_modules_transition1_modules_0_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_transition1_modules_0_modules_1_buffers_running_var_
        ) = (
            l_self_modules_transition1_modules_0_modules_1_parameters_weight_
        ) = l_self_modules_transition1_modules_0_modules_1_parameters_bias_ = None
        input_5 = torch.nn.functional.relu(input_4, inplace=True)
        input_4 = None
        input_6 = torch.conv2d(
            x_15,
            l_self_modules_transition1_modules_1_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_15 = (
            l_self_modules_transition1_modules_1_modules_0_modules_0_parameters_weight_
        ) = None
        input_7 = torch.nn.functional.batch_norm(
            input_6,
            l_self_modules_transition1_modules_1_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_transition1_modules_1_modules_0_modules_1_buffers_running_var_,
            l_self_modules_transition1_modules_1_modules_0_modules_1_parameters_weight_,
            l_self_modules_transition1_modules_1_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_6 = l_self_modules_transition1_modules_1_modules_0_modules_1_buffers_running_mean_ = l_self_modules_transition1_modules_1_modules_0_modules_1_buffers_running_var_ = (
            l_self_modules_transition1_modules_1_modules_0_modules_1_parameters_weight_
        ) = (
            l_self_modules_transition1_modules_1_modules_0_modules_1_parameters_bias_
        ) = None
        input_8 = torch.nn.functional.relu(input_7, inplace=True)
        input_7 = None
        x_16 = torch.conv2d(
            input_5,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = (None)
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = (None)
        x_20 += input_5
        x_21 = x_20
        x_20 = input_5 = None
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = (None)
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = (None)
        x_27 += x_22
        x_28 = x_27
        x_27 = x_22 = None
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_30 = torch.conv2d(
            input_8,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = (None)
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = (None)
        x_34 += input_8
        x_35 = x_34
        x_34 = input_8 = None
        x_36 = torch.nn.functional.relu(x_35, inplace=True)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = (None)
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = (None)
        x_41 += x_36
        x_42 = x_41
        x_41 = x_36 = None
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        input_9 = torch.conv2d(
            x_43,
            l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = (
            None
        )
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage2_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_11 = torch.nn.functional.interpolate(
            input_10, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_10 = None
        y = x_29 + input_11
        input_11 = None
        shortcut = torch.nn.functional.relu(y, inplace=False)
        y = None
        input_12 = torch.conv2d(
            x_29,
            l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        y_1 = input_13 + x_43
        input_13 = x_43 = None
        shortcut_1 = torch.nn.functional.relu(y_1, inplace=False)
        y_1 = None
        input_14 = torch.conv2d(
            shortcut_1,
            l_self_modules_transition2_modules_2_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_transition2_modules_2_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_transition2_modules_2_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_transition2_modules_2_modules_0_modules_1_buffers_running_var_,
            l_self_modules_transition2_modules_2_modules_0_modules_1_parameters_weight_,
            l_self_modules_transition2_modules_2_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_transition2_modules_2_modules_0_modules_1_buffers_running_mean_ = l_self_modules_transition2_modules_2_modules_0_modules_1_buffers_running_var_ = (
            l_self_modules_transition2_modules_2_modules_0_modules_1_parameters_weight_
        ) = (
            l_self_modules_transition2_modules_2_modules_0_modules_1_parameters_bias_
        ) = None
        input_16 = torch.nn.functional.relu(input_15, inplace=True)
        input_15 = None
        x_44 = torch.conv2d(
            shortcut,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = (None)
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = (None)
        x_48 += shortcut
        x_49 = x_48
        x_48 = shortcut = None
        x_50 = torch.nn.functional.relu(x_49, inplace=True)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = (None)
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = (None)
        x_55 += x_50
        x_56 = x_55
        x_55 = x_50 = None
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            shortcut_1,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = (None)
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = (None)
        x_62 += shortcut_1
        x_63 = x_62
        x_62 = shortcut_1 = None
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = (None)
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = (None)
        x_69 += x_64
        x_70 = x_69
        x_69 = x_64 = None
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        x_72 = torch.conv2d(
            input_16,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = (None)
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = (None)
        x_76 += input_16
        x_77 = x_76
        x_76 = input_16 = None
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = (None)
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = (None)
        x_83 += x_78
        x_84 = x_83
        x_83 = x_78 = None
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        input_17 = torch.conv2d(
            x_71,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = (
            None
        )
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_19 = torch.nn.functional.interpolate(
            input_18, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_18 = None
        y_2 = x_57 + input_19
        input_19 = None
        input_20 = torch.conv2d(
            x_85,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_ = (
            None
        )
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_20 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_ = (None)
        input_22 = torch.nn.functional.interpolate(
            input_21, None, 4.0, "nearest", None, recompute_scale_factor=None
        )
        input_21 = None
        y_3 = y_2 + input_22
        y_2 = input_22 = None
        shortcut_2 = torch.nn.functional.relu(y_3, inplace=False)
        y_3 = None
        input_23 = torch.conv2d(
            x_57,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_23 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        y_4 = input_24 + x_71
        input_24 = None
        input_25 = torch.conv2d(
            x_85,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_ = (
            None
        )
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_25 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_ = (None)
        input_27 = torch.nn.functional.interpolate(
            input_26, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_26 = None
        y_5 = y_4 + input_27
        y_4 = input_27 = None
        shortcut_3 = torch.nn.functional.relu(y_5, inplace=False)
        y_5 = None
        input_28 = torch.conv2d(
            x_57,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_ = (None)
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_28 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        input_30 = torch.nn.functional.relu(input_29, inplace=False)
        input_29 = None
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_30 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_ = (None)
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_31 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_33 = torch.conv2d(
            x_71,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_ = (None)
        input_34 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_33 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_ = (None)
        y_6 = input_32 + input_34
        input_32 = input_34 = None
        y_7 = y_6 + x_85
        y_6 = x_85 = None
        shortcut_4 = torch.nn.functional.relu(y_7, inplace=False)
        y_7 = None
        input_35 = torch.conv2d(
            shortcut_4,
            l_self_modules_transition3_modules_3_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_transition3_modules_3_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_var_,
            l_self_modules_transition3_modules_3_modules_0_modules_1_parameters_weight_,
            l_self_modules_transition3_modules_3_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_35 = l_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_mean_ = l_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_var_ = (
            l_self_modules_transition3_modules_3_modules_0_modules_1_parameters_weight_
        ) = (
            l_self_modules_transition3_modules_3_modules_0_modules_1_parameters_bias_
        ) = None
        input_37 = torch.nn.functional.relu(input_36, inplace=True)
        input_36 = None
        x_86 = torch.conv2d(
            shortcut_2,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = (None)
        x_88 = torch.nn.functional.relu(x_87, inplace=True)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = (None)
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = (None)
        x_90 += shortcut_2
        x_91 = x_90
        x_90 = shortcut_2 = None
        x_92 = torch.nn.functional.relu(x_91, inplace=True)
        x_91 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = (None)
        x_95 = torch.nn.functional.relu(x_94, inplace=True)
        x_94 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_95 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = (None)
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = (None)
        x_97 += x_92
        x_98 = x_97
        x_97 = x_92 = None
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        x_100 = torch.conv2d(
            shortcut_3,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = (None)
        x_102 = torch.nn.functional.relu(x_101, inplace=True)
        x_101 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_102 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = (None)
        x_104 += shortcut_3
        x_105 = x_104
        x_104 = shortcut_3 = None
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = (None)
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = (None)
        x_111 += x_106
        x_112 = x_111
        x_111 = x_106 = None
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        x_114 = torch.conv2d(
            shortcut_4,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = (None)
        x_116 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = (None)
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = (None)
        x_118 += shortcut_4
        x_119 = x_118
        x_118 = shortcut_4 = None
        x_120 = torch.nn.functional.relu(x_119, inplace=True)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = (None)
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_123 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = (None)
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = (None)
        x_125 += x_120
        x_126 = x_125
        x_125 = x_120 = None
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        x_128 = torch.conv2d(
            input_37,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_ = (None)
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_ = (None)
        x_132 += input_37
        x_133 = x_132
        x_132 = input_37 = None
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_ = (None)
        x_137 = torch.nn.functional.relu(x_136, inplace=True)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_ = (None)
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_ = (None)
        x_139 += x_134
        x_140 = x_139
        x_139 = x_134 = None
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        input_38 = torch.conv2d(
            x_113,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = (
            None
        )
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_38 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.interpolate(
            input_39, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_39 = None
        y_8 = x_99 + input_40
        input_40 = None
        input_41 = torch.conv2d(
            x_127,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_ = (
            None
        )
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_41 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_ = (None)
        input_43 = torch.nn.functional.interpolate(
            input_42, None, 4.0, "nearest", None, recompute_scale_factor=None
        )
        input_42 = None
        y_9 = y_8 + input_43
        y_8 = input_43 = None
        input_44 = torch.conv2d(
            x_141,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_0_parameters_weight_ = (
            None
        )
        input_45 = torch.nn.functional.batch_norm(
            input_44,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_44 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_ = (None)
        input_46 = torch.nn.functional.interpolate(
            input_45, None, 8.0, "nearest", None, recompute_scale_factor=None
        )
        input_45 = None
        y_10 = y_9 + input_46
        y_9 = input_46 = None
        shortcut_5 = torch.nn.functional.relu(y_10, inplace=False)
        y_10 = None
        input_47 = torch.conv2d(
            x_99,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_48 = torch.nn.functional.batch_norm(
            input_47,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_47 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        y_11 = input_48 + x_113
        input_48 = None
        input_49 = torch.conv2d(
            x_127,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_ = (
            None
        )
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_49 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_ = (None)
        input_51 = torch.nn.functional.interpolate(
            input_50, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_50 = None
        y_12 = y_11 + input_51
        y_11 = input_51 = None
        input_52 = torch.conv2d(
            x_141,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_0_parameters_weight_ = (
            None
        )
        input_53 = torch.nn.functional.batch_norm(
            input_52,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_52 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_ = (None)
        input_54 = torch.nn.functional.interpolate(
            input_53, None, 4.0, "nearest", None, recompute_scale_factor=None
        )
        input_53 = None
        y_13 = y_12 + input_54
        y_12 = input_54 = None
        shortcut_6 = torch.nn.functional.relu(y_13, inplace=False)
        y_13 = None
        input_55 = torch.conv2d(
            x_99,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_55 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        input_57 = torch.nn.functional.relu(input_56, inplace=False)
        input_56 = None
        input_58 = torch.conv2d(
            input_57,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_57 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_ = (None)
        input_59 = torch.nn.functional.batch_norm(
            input_58,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_58 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_60 = torch.conv2d(
            x_113,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_61 = torch.nn.functional.batch_norm(
            input_60,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_60 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_ = (None)
        y_14 = input_59 + input_61
        input_59 = input_61 = None
        y_15 = y_14 + x_127
        y_14 = None
        input_62 = torch.conv2d(
            x_141,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_0_parameters_weight_ = (
            None
        )
        input_63 = torch.nn.functional.batch_norm(
            input_62,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_62 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_ = (None)
        input_64 = torch.nn.functional.interpolate(
            input_63, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_63 = None
        y_16 = y_15 + input_64
        y_15 = input_64 = None
        shortcut_7 = torch.nn.functional.relu(y_16, inplace=False)
        y_16 = None
        input_65 = torch.conv2d(
            x_99,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_ = (None)
        input_66 = torch.nn.functional.batch_norm(
            input_65,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_65 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        input_67 = torch.nn.functional.relu(input_66, inplace=False)
        input_66 = None
        input_68 = torch.conv2d(
            input_67,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_67 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_ = (None)
        input_69 = torch.nn.functional.batch_norm(
            input_68,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_68 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_70 = torch.nn.functional.relu(input_69, inplace=False)
        input_69 = None
        input_71 = torch.conv2d(
            input_70,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_70 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_ = (None)
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_71 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_ = (None)
        input_73 = torch.conv2d(
            x_113,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_ = (None)
        input_74 = torch.nn.functional.batch_norm(
            input_73,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_73 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_ = (None)
        input_75 = torch.nn.functional.relu(input_74, inplace=False)
        input_74 = None
        input_76 = torch.conv2d(
            input_75,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_75 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_ = (None)
        input_77 = torch.nn.functional.batch_norm(
            input_76,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_76 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_ = (None)
        y_17 = input_72 + input_77
        input_72 = input_77 = None
        input_78 = torch.conv2d(
            x_127,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_ = (None)
        input_79 = torch.nn.functional.batch_norm(
            input_78,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_78 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_ = (None)
        y_18 = y_17 + input_79
        y_17 = input_79 = None
        y_19 = y_18 + x_141
        y_18 = x_141 = None
        shortcut_8 = torch.nn.functional.relu(y_19, inplace=False)
        y_19 = None
        x_142 = torch.conv2d(
            shortcut_5,
            l_self_modules_incre_modules_modules_0_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_incre_modules_modules_0_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_weight_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_bias_ = (None)
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_incre_modules_modules_0_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_144 = l_self_modules_incre_modules_modules_0_modules_0_modules_conv2_parameters_weight_ = (None)
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_weight_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_bias_ = (None)
        x_147 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_incre_modules_modules_0_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_incre_modules_modules_0_modules_0_modules_conv3_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_weight_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_bias_ = (None)
        input_80 = torch.conv2d(
            shortcut_5,
            l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        shortcut_5 = l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_81 = torch.nn.functional.batch_norm(
            input_80,
            l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_80 = l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_149 += input_81
        x_150 = x_149
        x_149 = input_81 = None
        x_151 = torch.nn.functional.relu(x_150, inplace=True)
        x_150 = None
        x_152 = torch.conv2d(
            shortcut_6,
            l_self_modules_incre_modules_modules_1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_incre_modules_modules_1_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_bias_ = (None)
        x_154 = torch.nn.functional.relu(x_153, inplace=True)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_incre_modules_modules_1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_incre_modules_modules_1_modules_0_modules_conv2_parameters_weight_ = (None)
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_bias_ = (None)
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_incre_modules_modules_1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_incre_modules_modules_1_modules_0_modules_conv3_parameters_weight_ = (None)
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_weight_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_bias_ = (None)
        input_82 = torch.conv2d(
            shortcut_6,
            l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        shortcut_6 = l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_83 = torch.nn.functional.batch_norm(
            input_82,
            l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_82 = l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_159 += input_83
        x_160 = x_159
        x_159 = input_83 = None
        x_161 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        input_84 = torch.conv2d(
            x_151,
            l_self_modules_downsamp_modules_modules_0_modules_0_parameters_weight_,
            l_self_modules_downsamp_modules_modules_0_modules_0_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_151 = (
            l_self_modules_downsamp_modules_modules_0_modules_0_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_0_modules_0_parameters_bias_ = None
        input_85 = torch.nn.functional.batch_norm(
            input_84,
            l_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_var_,
            l_self_modules_downsamp_modules_modules_0_modules_1_parameters_weight_,
            l_self_modules_downsamp_modules_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_84 = (
            l_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_var_
        ) = (
            l_self_modules_downsamp_modules_modules_0_modules_1_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_0_modules_1_parameters_bias_ = None
        input_86 = torch.nn.functional.relu(input_85, inplace=True)
        input_85 = None
        y_20 = x_161 + input_86
        x_161 = input_86 = None
        x_162 = torch.conv2d(
            shortcut_7,
            l_self_modules_incre_modules_modules_2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_incre_modules_modules_2_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_bias_ = (None)
        x_164 = torch.nn.functional.relu(x_163, inplace=True)
        x_163 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_incre_modules_modules_2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_164 = l_self_modules_incre_modules_modules_2_modules_0_modules_conv2_parameters_weight_ = (None)
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_165 = l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_bias_ = (None)
        x_167 = torch.nn.functional.relu(x_166, inplace=True)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_incre_modules_modules_2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_incre_modules_modules_2_modules_0_modules_conv3_parameters_weight_ = (None)
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_weight_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_bias_ = (None)
        input_87 = torch.conv2d(
            shortcut_7,
            l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        shortcut_7 = l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_88 = torch.nn.functional.batch_norm(
            input_87,
            l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_87 = l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_169 += input_88
        x_170 = x_169
        x_169 = input_88 = None
        x_171 = torch.nn.functional.relu(x_170, inplace=True)
        x_170 = None
        input_89 = torch.conv2d(
            y_20,
            l_self_modules_downsamp_modules_modules_1_modules_0_parameters_weight_,
            l_self_modules_downsamp_modules_modules_1_modules_0_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        y_20 = (
            l_self_modules_downsamp_modules_modules_1_modules_0_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_1_modules_0_parameters_bias_ = None
        input_90 = torch.nn.functional.batch_norm(
            input_89,
            l_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_var_,
            l_self_modules_downsamp_modules_modules_1_modules_1_parameters_weight_,
            l_self_modules_downsamp_modules_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_89 = (
            l_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_downsamp_modules_modules_1_modules_1_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_1_modules_1_parameters_bias_ = None
        input_91 = torch.nn.functional.relu(input_90, inplace=True)
        input_90 = None
        y_21 = x_171 + input_91
        x_171 = input_91 = None
        x_172 = torch.conv2d(
            shortcut_8,
            l_self_modules_incre_modules_modules_3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_incre_modules_modules_3_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_bias_ = (None)
        x_174 = torch.nn.functional.relu(x_173, inplace=True)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_incre_modules_modules_3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_incre_modules_modules_3_modules_0_modules_conv2_parameters_weight_ = (None)
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_bias_ = (None)
        x_177 = torch.nn.functional.relu(x_176, inplace=True)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_incre_modules_modules_3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_incre_modules_modules_3_modules_0_modules_conv3_parameters_weight_ = (None)
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_weight_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_bias_ = (None)
        input_92 = torch.conv2d(
            shortcut_8,
            l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        shortcut_8 = l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_93 = torch.nn.functional.batch_norm(
            input_92,
            l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_92 = l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_179 += input_93
        x_180 = x_179
        x_179 = input_93 = None
        x_181 = torch.nn.functional.relu(x_180, inplace=True)
        x_180 = None
        input_94 = torch.conv2d(
            y_21,
            l_self_modules_downsamp_modules_modules_2_modules_0_parameters_weight_,
            l_self_modules_downsamp_modules_modules_2_modules_0_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        y_21 = (
            l_self_modules_downsamp_modules_modules_2_modules_0_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_2_modules_0_parameters_bias_ = None
        input_95 = torch.nn.functional.batch_norm(
            input_94,
            l_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_var_,
            l_self_modules_downsamp_modules_modules_2_modules_1_parameters_weight_,
            l_self_modules_downsamp_modules_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_94 = (
            l_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_var_
        ) = (
            l_self_modules_downsamp_modules_modules_2_modules_1_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_2_modules_1_parameters_bias_ = None
        input_96 = torch.nn.functional.relu(input_95, inplace=True)
        input_95 = None
        y_22 = x_181 + input_96
        x_181 = input_96 = None
        input_97 = torch.conv2d(
            y_22,
            l_self_modules_final_layer_modules_0_parameters_weight_,
            l_self_modules_final_layer_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        y_22 = (
            l_self_modules_final_layer_modules_0_parameters_weight_
        ) = l_self_modules_final_layer_modules_0_parameters_bias_ = None
        input_98 = torch.nn.functional.batch_norm(
            input_97,
            l_self_modules_final_layer_modules_1_buffers_running_mean_,
            l_self_modules_final_layer_modules_1_buffers_running_var_,
            l_self_modules_final_layer_modules_1_parameters_weight_,
            l_self_modules_final_layer_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_97 = (
            l_self_modules_final_layer_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_final_layer_modules_1_buffers_running_var_
        ) = (
            l_self_modules_final_layer_modules_1_parameters_weight_
        ) = l_self_modules_final_layer_modules_1_parameters_bias_ = None
        input_99 = torch.nn.functional.relu(input_98, inplace=True)
        input_98 = None
        x_182 = torch.nn.functional.adaptive_avg_pool2d(input_99, 1)
        input_99 = None
        x_183 = x_182.flatten(1, -1)
        x_182 = None
        x_184 = torch.nn.functional.dropout(x_183, 0.0, False, False)
        x_183 = None
        x_185 = torch._C._nn.linear(
            x_184,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_184 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_185,)
