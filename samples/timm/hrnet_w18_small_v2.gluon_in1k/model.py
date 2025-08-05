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
        L_self_modules_layer1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bn3_parameters_bias_
        )
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
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_ = L_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_ = L_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_
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
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_ = L_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_ = L_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_
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
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_ = None
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_ = None
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_layer1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_layer1_modules_1_modules_conv2_parameters_weight_ = None
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = (
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn2_parameters_bias_ = None
        x_21 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_ = None
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_ = None
        x_23 += x_15
        x_24 = x_23
        x_23 = x_15 = None
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        input_3 = torch.conv2d(
            x_25,
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
            x_25,
            l_self_modules_transition1_modules_1_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_25 = (
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
        x_26 = torch.conv2d(
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
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = (None)
        x_28 = torch.nn.functional.relu(x_27, inplace=True)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = (None)
        x_30 += input_5
        x_31 = x_30
        x_30 = input_5 = None
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x_33 = torch.conv2d(
            x_32,
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
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = (None)
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = (None)
        x_37 += x_32
        x_38 = x_37
        x_37 = x_32 = None
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        x_40 = torch.conv2d(
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
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = (None)
        x_44 += input_8
        x_45 = x_44
        x_44 = input_8 = None
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
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
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = (None)
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage2_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = (None)
        x_51 += x_46
        x_52 = x_51
        x_51 = x_46 = None
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        input_9 = torch.conv2d(
            x_53,
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
        y = x_39 + input_11
        input_11 = None
        shortcut = torch.nn.functional.relu(y, inplace=False)
        y = None
        input_12 = torch.conv2d(
            x_39,
            l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_stage2_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = (None)
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
        y_1 = input_13 + x_53
        input_13 = x_53 = None
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
        x_54 = torch.conv2d(
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
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = (None)
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_56 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = (None)
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = (None)
        x_58 += shortcut
        x_59 = x_58
        x_58 = shortcut = None
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
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
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = (None)
        x_63 = torch.nn.functional.relu(x_62, inplace=True)
        x_62 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = (None)
        x_65 += x_60
        x_66 = x_65
        x_65 = x_60 = None
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
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
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = (None)
        x_70 = torch.nn.functional.relu(x_69, inplace=True)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = (None)
        x_72 += shortcut_1
        x_73 = x_72
        x_72 = shortcut_1 = None
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
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
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = (None)
        x_77 = torch.nn.functional.relu(x_76, inplace=True)
        x_76 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_77 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = (None)
        x_79 += x_74
        x_80 = x_79
        x_79 = x_74 = None
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
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
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = (None)
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = (None)
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = (None)
        x_86 += input_16
        x_87 = x_86
        x_86 = input_16 = None
        x_88 = torch.nn.functional.relu(x_87, inplace=True)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
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
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = (None)
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = (None)
        x_93 += x_88
        x_94 = x_93
        x_93 = x_88 = None
        x_95 = torch.nn.functional.relu(x_94, inplace=True)
        x_94 = None
        input_17 = torch.conv2d(
            x_81,
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
        y_2 = x_67 + input_19
        input_19 = None
        input_20 = torch.conv2d(
            x_95,
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
            x_67,
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
        y_4 = input_24 + x_81
        input_24 = None
        input_25 = torch.conv2d(
            x_95,
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
            x_67,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_ = (None)
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
            x_81,
            l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_stage3_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_ = (None)
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
        y_7 = y_6 + x_95
        y_6 = x_95 = None
        shortcut_4 = torch.nn.functional.relu(y_7, inplace=False)
        y_7 = None
        x_96 = torch.conv2d(
            shortcut_2,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = (None)
        x_98 = torch.nn.functional.relu(x_97, inplace=True)
        x_97 = None
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_98 = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = (None)
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = (None)
        x_100 += shortcut_2
        x_101 = x_100
        x_100 = shortcut_2 = None
        x_102 = torch.nn.functional.relu(x_101, inplace=True)
        x_101 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = (None)
        x_105 = torch.nn.functional.relu(x_104, inplace=True)
        x_104 = None
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_105 = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = (None)
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = (None)
        x_107 += x_102
        x_108 = x_107
        x_107 = x_102 = None
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            shortcut_3,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = (None)
        x_114 += shortcut_3
        x_115 = x_114
        x_114 = shortcut_3 = None
        x_116 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = (None)
        x_119 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = (None)
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = (None)
        x_121 += x_116
        x_122 = x_121
        x_121 = x_116 = None
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            shortcut_4,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = (None)
        x_126 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_126 = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = (None)
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = (None)
        x_128 += shortcut_4
        x_129 = x_128
        x_128 = shortcut_4 = None
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = (None)
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = (None)
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = (None)
        x_135 += x_130
        x_136 = x_135
        x_135 = x_130 = None
        x_137 = torch.nn.functional.relu(x_136, inplace=True)
        x_136 = None
        input_35 = torch.conv2d(
            x_123,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = (
            None
        )
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_35 = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_37 = torch.nn.functional.interpolate(
            input_36, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_36 = None
        y_8 = x_109 + input_37
        input_37 = None
        input_38 = torch.conv2d(
            x_137,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_ = (
            None
        )
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_38 = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.interpolate(
            input_39, None, 4.0, "nearest", None, recompute_scale_factor=None
        )
        input_39 = None
        y_9 = y_8 + input_40
        y_8 = input_40 = None
        shortcut_5 = torch.nn.functional.relu(y_9, inplace=False)
        y_9 = None
        input_41 = torch.conv2d(
            x_109,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_41 = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        y_10 = input_42 + x_123
        input_42 = None
        input_43 = torch.conv2d(
            x_137,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_ = (
            None
        )
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_43 = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.interpolate(
            input_44, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_44 = None
        y_11 = y_10 + input_45
        y_10 = input_45 = None
        shortcut_6 = torch.nn.functional.relu(y_11, inplace=False)
        y_11 = None
        input_46 = torch.conv2d(
            x_109,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_ = (None)
        input_47 = torch.nn.functional.batch_norm(
            input_46,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_46 = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        input_48 = torch.nn.functional.relu(input_47, inplace=False)
        input_47 = None
        input_49 = torch.conv2d(
            input_48,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_48 = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_ = (None)
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_49 = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_51 = torch.conv2d(
            x_123,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_123 = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_ = (None)
        input_52 = torch.nn.functional.batch_norm(
            input_51,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_51 = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_ = l_self_modules_stage3_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_ = (None)
        y_12 = input_50 + input_52
        input_50 = input_52 = None
        y_13 = y_12 + x_137
        y_12 = x_137 = None
        shortcut_7 = torch.nn.functional.relu(y_13, inplace=False)
        y_13 = None
        x_138 = torch.conv2d(
            shortcut_5,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = (None)
        x_140 = torch.nn.functional.relu(x_139, inplace=True)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = (None)
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = (None)
        x_142 += shortcut_5
        x_143 = x_142
        x_142 = shortcut_5 = None
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = (None)
        x_147 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = (None)
        x_149 += x_144
        x_150 = x_149
        x_149 = x_144 = None
        x_151 = torch.nn.functional.relu(x_150, inplace=True)
        x_150 = None
        x_152 = torch.conv2d(
            shortcut_6,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = (None)
        x_154 = torch.nn.functional.relu(x_153, inplace=True)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = (None)
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = (None)
        x_156 += shortcut_6
        x_157 = x_156
        x_156 = shortcut_6 = None
        x_158 = torch.nn.functional.relu(x_157, inplace=True)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = (None)
        x_161 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = (None)
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = (None)
        x_163 += x_158
        x_164 = x_163
        x_163 = x_158 = None
        x_165 = torch.nn.functional.relu(x_164, inplace=True)
        x_164 = None
        x_166 = torch.conv2d(
            shortcut_7,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = (None)
        x_168 = torch.nn.functional.relu(x_167, inplace=True)
        x_167 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = (None)
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_169 = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = (None)
        x_170 += shortcut_7
        x_171 = x_170
        x_170 = shortcut_7 = None
        x_172 = torch.nn.functional.relu(x_171, inplace=True)
        x_171 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = (None)
        x_175 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = (None)
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage3_modules_2_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = (None)
        x_177 += x_172
        x_178 = x_177
        x_177 = x_172 = None
        x_179 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        input_53 = torch.conv2d(
            x_165,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = (
            None
        )
        input_54 = torch.nn.functional.batch_norm(
            input_53,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_53 = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_55 = torch.nn.functional.interpolate(
            input_54, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_54 = None
        y_14 = x_151 + input_55
        input_55 = None
        input_56 = torch.conv2d(
            x_179,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_ = (
            None
        )
        input_57 = torch.nn.functional.batch_norm(
            input_56,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_56 = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_ = (None)
        input_58 = torch.nn.functional.interpolate(
            input_57, None, 4.0, "nearest", None, recompute_scale_factor=None
        )
        input_57 = None
        y_15 = y_14 + input_58
        y_14 = input_58 = None
        shortcut_8 = torch.nn.functional.relu(y_15, inplace=False)
        y_15 = None
        input_59 = torch.conv2d(
            x_151,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_60 = torch.nn.functional.batch_norm(
            input_59,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_59 = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        y_16 = input_60 + x_165
        input_60 = None
        input_61 = torch.conv2d(
            x_179,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_ = (
            None
        )
        input_62 = torch.nn.functional.batch_norm(
            input_61,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_61 = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_ = (None)
        input_63 = torch.nn.functional.interpolate(
            input_62, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_62 = None
        y_17 = y_16 + input_63
        y_16 = input_63 = None
        shortcut_9 = torch.nn.functional.relu(y_17, inplace=False)
        y_17 = None
        input_64 = torch.conv2d(
            x_151,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_ = (None)
        input_65 = torch.nn.functional.batch_norm(
            input_64,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_64 = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        input_66 = torch.nn.functional.relu(input_65, inplace=False)
        input_65 = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_66 = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_ = (None)
        input_68 = torch.nn.functional.batch_norm(
            input_67,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_67 = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_69 = torch.conv2d(
            x_165,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_165 = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_ = (None)
        input_70 = torch.nn.functional.batch_norm(
            input_69,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_69 = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_ = l_self_modules_stage3_modules_2_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_ = (None)
        y_18 = input_68 + input_70
        input_68 = input_70 = None
        y_19 = y_18 + x_179
        y_18 = x_179 = None
        shortcut_10 = torch.nn.functional.relu(y_19, inplace=False)
        y_19 = None
        input_71 = torch.conv2d(
            shortcut_10,
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
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_var_,
            l_self_modules_transition3_modules_3_modules_0_modules_1_parameters_weight_,
            l_self_modules_transition3_modules_3_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_71 = l_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_mean_ = l_self_modules_transition3_modules_3_modules_0_modules_1_buffers_running_var_ = (
            l_self_modules_transition3_modules_3_modules_0_modules_1_parameters_weight_
        ) = (
            l_self_modules_transition3_modules_3_modules_0_modules_1_parameters_bias_
        ) = None
        input_73 = torch.nn.functional.relu(input_72, inplace=True)
        input_72 = None
        x_180 = torch.conv2d(
            shortcut_8,
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
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = (None)
        x_182 = torch.nn.functional.relu(x_181, inplace=True)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = (None)
        x_184 += shortcut_8
        x_185 = x_184
        x_184 = shortcut_8 = None
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        x_187 = torch.conv2d(
            x_186,
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
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = (None)
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_189 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = (None)
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = (None)
        x_191 += x_186
        x_192 = x_191
        x_191 = x_186 = None
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        x_194 = torch.conv2d(
            shortcut_9,
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
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = (None)
        x_196 = torch.nn.functional.relu(x_195, inplace=True)
        x_195 = None
        x_197 = torch.conv2d(
            x_196,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_196 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = (None)
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_197 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = (None)
        x_198 += shortcut_9
        x_199 = x_198
        x_198 = shortcut_9 = None
        x_200 = torch.nn.functional.relu(x_199, inplace=True)
        x_199 = None
        x_201 = torch.conv2d(
            x_200,
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
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_201 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = (None)
        x_203 = torch.nn.functional.relu(x_202, inplace=True)
        x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_203 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = (None)
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = (None)
        x_205 += x_200
        x_206 = x_205
        x_205 = x_200 = None
        x_207 = torch.nn.functional.relu(x_206, inplace=True)
        x_206 = None
        x_208 = torch.conv2d(
            shortcut_10,
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
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_208 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = (None)
        x_210 = torch.nn.functional.relu(x_209, inplace=True)
        x_209 = None
        x_211 = torch.conv2d(
            x_210,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_210 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = (None)
        x_212 = torch.nn.functional.batch_norm(
            x_211,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_211 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = (None)
        x_212 += shortcut_10
        x_213 = x_212
        x_212 = shortcut_10 = None
        x_214 = torch.nn.functional.relu(x_213, inplace=True)
        x_213 = None
        x_215 = torch.conv2d(
            x_214,
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
        x_216 = torch.nn.functional.batch_norm(
            x_215,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_215 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = (None)
        x_217 = torch.nn.functional.relu(x_216, inplace=True)
        x_216 = None
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = (None)
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_218 = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = (None)
        x_219 += x_214
        x_220 = x_219
        x_219 = x_214 = None
        x_221 = torch.nn.functional.relu(x_220, inplace=True)
        x_220 = None
        x_222 = torch.conv2d(
            input_73,
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
        x_223 = torch.nn.functional.batch_norm(
            x_222,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_222 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_ = (None)
        x_224 = torch.nn.functional.relu(x_223, inplace=True)
        x_223 = None
        x_225 = torch.conv2d(
            x_224,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_224 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_ = (None)
        x_226 = torch.nn.functional.batch_norm(
            x_225,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_225 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_ = (None)
        x_226 += input_73
        x_227 = x_226
        x_226 = input_73 = None
        x_228 = torch.nn.functional.relu(x_227, inplace=True)
        x_227 = None
        x_229 = torch.conv2d(
            x_228,
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
        x_230 = torch.nn.functional.batch_norm(
            x_229,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_229 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_ = (None)
        x_231 = torch.nn.functional.relu(x_230, inplace=True)
        x_230 = None
        x_232 = torch.conv2d(
            x_231,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_231 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_ = (None)
        x_233 = torch.nn.functional.batch_norm(
            x_232,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_232 = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_0_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_ = (None)
        x_233 += x_228
        x_234 = x_233
        x_233 = x_228 = None
        x_235 = torch.nn.functional.relu(x_234, inplace=True)
        x_234 = None
        input_74 = torch.conv2d(
            x_207,
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
        input_75 = torch.nn.functional.batch_norm(
            input_74,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_74 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_76 = torch.nn.functional.interpolate(
            input_75, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_75 = None
        y_20 = x_193 + input_76
        input_76 = None
        input_77 = torch.conv2d(
            x_221,
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
        input_78 = torch.nn.functional.batch_norm(
            input_77,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_77 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_ = (None)
        input_79 = torch.nn.functional.interpolate(
            input_78, None, 4.0, "nearest", None, recompute_scale_factor=None
        )
        input_78 = None
        y_21 = y_20 + input_79
        y_20 = input_79 = None
        input_80 = torch.conv2d(
            x_235,
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
        input_81 = torch.nn.functional.batch_norm(
            input_80,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_80 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_ = (None)
        input_82 = torch.nn.functional.interpolate(
            input_81, None, 8.0, "nearest", None, recompute_scale_factor=None
        )
        input_81 = None
        y_22 = y_21 + input_82
        y_21 = input_82 = None
        shortcut_11 = torch.nn.functional.relu(y_22, inplace=False)
        y_22 = None
        input_83 = torch.conv2d(
            x_193,
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
        input_84 = torch.nn.functional.batch_norm(
            input_83,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_83 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        y_23 = input_84 + x_207
        input_84 = None
        input_85 = torch.conv2d(
            x_221,
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
        input_86 = torch.nn.functional.batch_norm(
            input_85,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_85 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_ = (None)
        input_87 = torch.nn.functional.interpolate(
            input_86, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_86 = None
        y_24 = y_23 + input_87
        y_23 = input_87 = None
        input_88 = torch.conv2d(
            x_235,
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
        input_89 = torch.nn.functional.batch_norm(
            input_88,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_88 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_ = (None)
        input_90 = torch.nn.functional.interpolate(
            input_89, None, 4.0, "nearest", None, recompute_scale_factor=None
        )
        input_89 = None
        y_25 = y_24 + input_90
        y_24 = input_90 = None
        shortcut_12 = torch.nn.functional.relu(y_25, inplace=False)
        y_25 = None
        input_91 = torch.conv2d(
            x_193,
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
        input_92 = torch.nn.functional.batch_norm(
            input_91,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_91 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        input_93 = torch.nn.functional.relu(input_92, inplace=False)
        input_92 = None
        input_94 = torch.conv2d(
            input_93,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_93 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_ = (None)
        input_95 = torch.nn.functional.batch_norm(
            input_94,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_94 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_96 = torch.conv2d(
            x_207,
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
        input_97 = torch.nn.functional.batch_norm(
            input_96,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_96 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_ = (None)
        y_26 = input_95 + input_97
        input_95 = input_97 = None
        y_27 = y_26 + x_221
        y_26 = None
        input_98 = torch.conv2d(
            x_235,
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
        input_99 = torch.nn.functional.batch_norm(
            input_98,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_98 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_ = (None)
        input_100 = torch.nn.functional.interpolate(
            input_99, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_99 = None
        y_28 = y_27 + input_100
        y_27 = input_100 = None
        shortcut_13 = torch.nn.functional.relu(y_28, inplace=False)
        y_28 = None
        input_101 = torch.conv2d(
            x_193,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_ = (None)
        input_102 = torch.nn.functional.batch_norm(
            input_101,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_101 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        input_103 = torch.nn.functional.relu(input_102, inplace=False)
        input_102 = None
        input_104 = torch.conv2d(
            input_103,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_103 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_ = (None)
        input_105 = torch.nn.functional.batch_norm(
            input_104,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_104 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_106 = torch.nn.functional.relu(input_105, inplace=False)
        input_105 = None
        input_107 = torch.conv2d(
            input_106,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_106 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_ = (None)
        input_108 = torch.nn.functional.batch_norm(
            input_107,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_107 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_ = (None)
        input_109 = torch.conv2d(
            x_207,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_207 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_ = (None)
        input_110 = torch.nn.functional.batch_norm(
            input_109,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_109 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_ = (None)
        input_111 = torch.nn.functional.relu(input_110, inplace=False)
        input_110 = None
        input_112 = torch.conv2d(
            input_111,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_111 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_ = (None)
        input_113 = torch.nn.functional.batch_norm(
            input_112,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_112 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_ = (None)
        y_29 = input_108 + input_113
        input_108 = input_113 = None
        input_114 = torch.conv2d(
            x_221,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_221 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_ = (None)
        input_115 = torch.nn.functional.batch_norm(
            input_114,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_114 = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_0_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_ = (None)
        y_30 = y_29 + input_115
        y_29 = input_115 = None
        y_31 = y_30 + x_235
        y_30 = x_235 = None
        shortcut_14 = torch.nn.functional.relu(y_31, inplace=False)
        y_31 = None
        x_236 = torch.conv2d(
            shortcut_11,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_236 = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn1_parameters_bias_ = (None)
        x_238 = torch.nn.functional.relu(x_237, inplace=True)
        x_237 = None
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_238 = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_conv2_parameters_weight_ = (None)
        x_240 = torch.nn.functional.batch_norm(
            x_239,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_239 = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_0_modules_bn2_parameters_bias_ = (None)
        x_240 += shortcut_11
        x_241 = x_240
        x_240 = shortcut_11 = None
        x_242 = torch.nn.functional.relu(x_241, inplace=True)
        x_241 = None
        x_243 = torch.conv2d(
            x_242,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_244 = torch.nn.functional.batch_norm(
            x_243,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_243 = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn1_parameters_bias_ = (None)
        x_245 = torch.nn.functional.relu(x_244, inplace=True)
        x_244 = None
        x_246 = torch.conv2d(
            x_245,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_245 = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_conv2_parameters_weight_ = (None)
        x_247 = torch.nn.functional.batch_norm(
            x_246,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_246 = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_0_modules_1_modules_bn2_parameters_bias_ = (None)
        x_247 += x_242
        x_248 = x_247
        x_247 = x_242 = None
        x_249 = torch.nn.functional.relu(x_248, inplace=True)
        x_248 = None
        x_250 = torch.conv2d(
            shortcut_12,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_251 = torch.nn.functional.batch_norm(
            x_250,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_250 = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn1_parameters_bias_ = (None)
        x_252 = torch.nn.functional.relu(x_251, inplace=True)
        x_251 = None
        x_253 = torch.conv2d(
            x_252,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_252 = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_conv2_parameters_weight_ = (None)
        x_254 = torch.nn.functional.batch_norm(
            x_253,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_253 = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_0_modules_bn2_parameters_bias_ = (None)
        x_254 += shortcut_12
        x_255 = x_254
        x_254 = shortcut_12 = None
        x_256 = torch.nn.functional.relu(x_255, inplace=True)
        x_255 = None
        x_257 = torch.conv2d(
            x_256,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_258 = torch.nn.functional.batch_norm(
            x_257,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_257 = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn1_parameters_bias_ = (None)
        x_259 = torch.nn.functional.relu(x_258, inplace=True)
        x_258 = None
        x_260 = torch.conv2d(
            x_259,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_259 = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_conv2_parameters_weight_ = (None)
        x_261 = torch.nn.functional.batch_norm(
            x_260,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_260 = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_1_modules_1_modules_bn2_parameters_bias_ = (None)
        x_261 += x_256
        x_262 = x_261
        x_261 = x_256 = None
        x_263 = torch.nn.functional.relu(x_262, inplace=True)
        x_262 = None
        x_264 = torch.conv2d(
            shortcut_13,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_265 = torch.nn.functional.batch_norm(
            x_264,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_264 = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn1_parameters_bias_ = (None)
        x_266 = torch.nn.functional.relu(x_265, inplace=True)
        x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_266 = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_conv2_parameters_weight_ = (None)
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_267 = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_0_modules_bn2_parameters_bias_ = (None)
        x_268 += shortcut_13
        x_269 = x_268
        x_268 = shortcut_13 = None
        x_270 = torch.nn.functional.relu(x_269, inplace=True)
        x_269 = None
        x_271 = torch.conv2d(
            x_270,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_272 = torch.nn.functional.batch_norm(
            x_271,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_271 = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn1_parameters_bias_ = (None)
        x_273 = torch.nn.functional.relu(x_272, inplace=True)
        x_272 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_273 = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_conv2_parameters_weight_ = (None)
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_274 = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_2_modules_1_modules_bn2_parameters_bias_ = (None)
        x_275 += x_270
        x_276 = x_275
        x_275 = x_270 = None
        x_277 = torch.nn.functional.relu(x_276, inplace=True)
        x_276 = None
        x_278 = torch.conv2d(
            shortcut_14,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        x_279 = torch.nn.functional.batch_norm(
            x_278,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_278 = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn1_parameters_bias_ = (None)
        x_280 = torch.nn.functional.relu(x_279, inplace=True)
        x_279 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_280 = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_conv2_parameters_weight_ = (None)
        x_282 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_281 = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_0_modules_bn2_parameters_bias_ = (None)
        x_282 += shortcut_14
        x_283 = x_282
        x_282 = shortcut_14 = None
        x_284 = torch.nn.functional.relu(x_283, inplace=True)
        x_283 = None
        x_285 = torch.conv2d(
            x_284,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        x_286 = torch.nn.functional.batch_norm(
            x_285,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_285 = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn1_parameters_bias_ = (None)
        x_287 = torch.nn.functional.relu(x_286, inplace=True)
        x_286 = None
        x_288 = torch.conv2d(
            x_287,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_287 = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_conv2_parameters_weight_ = (None)
        x_289 = torch.nn.functional.batch_norm(
            x_288,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_288 = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_parameters_weight_ = l_self_modules_stage4_modules_1_modules_branches_modules_3_modules_1_modules_bn2_parameters_bias_ = (None)
        x_289 += x_284
        x_290 = x_289
        x_289 = x_284 = None
        x_291 = torch.nn.functional.relu(x_290, inplace=True)
        x_290 = None
        input_116 = torch.conv2d(
            x_263,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_0_parameters_weight_ = (
            None
        )
        input_117 = torch.nn.functional.batch_norm(
            input_116,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_116 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_118 = torch.nn.functional.interpolate(
            input_117, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_117 = None
        y_32 = x_249 + input_118
        input_118 = None
        input_119 = torch.conv2d(
            x_277,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_0_parameters_weight_ = (
            None
        )
        input_120 = torch.nn.functional.batch_norm(
            input_119,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_119 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_2_modules_1_parameters_bias_ = (None)
        input_121 = torch.nn.functional.interpolate(
            input_120, None, 4.0, "nearest", None, recompute_scale_factor=None
        )
        input_120 = None
        y_33 = y_32 + input_121
        y_32 = input_121 = None
        input_122 = torch.conv2d(
            x_291,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_0_parameters_weight_ = (
            None
        )
        input_123 = torch.nn.functional.batch_norm(
            input_122,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_122 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_0_modules_3_modules_1_parameters_bias_ = (None)
        input_124 = torch.nn.functional.interpolate(
            input_123, None, 8.0, "nearest", None, recompute_scale_factor=None
        )
        input_123 = None
        y_34 = y_33 + input_124
        y_33 = input_124 = None
        shortcut_15 = torch.nn.functional.relu(y_34, inplace=False)
        y_34 = None
        input_125 = torch.conv2d(
            x_249,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_126 = torch.nn.functional.batch_norm(
            input_125,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_125 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        y_35 = input_126 + x_263
        input_126 = None
        input_127 = torch.conv2d(
            x_277,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_0_parameters_weight_ = (
            None
        )
        input_128 = torch.nn.functional.batch_norm(
            input_127,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_127 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_2_modules_1_parameters_bias_ = (None)
        input_129 = torch.nn.functional.interpolate(
            input_128, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_128 = None
        y_36 = y_35 + input_129
        y_35 = input_129 = None
        input_130 = torch.conv2d(
            x_291,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_0_parameters_weight_ = (
            None
        )
        input_131 = torch.nn.functional.batch_norm(
            input_130,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_130 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_1_modules_3_modules_1_parameters_bias_ = (None)
        input_132 = torch.nn.functional.interpolate(
            input_131, None, 4.0, "nearest", None, recompute_scale_factor=None
        )
        input_131 = None
        y_37 = y_36 + input_132
        y_36 = input_132 = None
        shortcut_16 = torch.nn.functional.relu(y_37, inplace=False)
        y_37 = None
        input_133 = torch.conv2d(
            x_249,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_134 = torch.nn.functional.batch_norm(
            input_133,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_133 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        input_135 = torch.nn.functional.relu(input_134, inplace=False)
        input_134 = None
        input_136 = torch.conv2d(
            input_135,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_135 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_0_parameters_weight_ = (None)
        input_137 = torch.nn.functional.batch_norm(
            input_136,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_136 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_138 = torch.conv2d(
            x_263,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_139 = torch.nn.functional.batch_norm(
            input_138,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_138 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_1_modules_0_modules_1_parameters_bias_ = (None)
        y_38 = input_137 + input_139
        input_137 = input_139 = None
        y_39 = y_38 + x_277
        y_38 = None
        input_140 = torch.conv2d(
            x_291,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_0_parameters_weight_ = (
            None
        )
        input_141 = torch.nn.functional.batch_norm(
            input_140,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_140 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_2_modules_3_modules_1_parameters_bias_ = (None)
        input_142 = torch.nn.functional.interpolate(
            input_141, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        input_141 = None
        y_40 = y_39 + input_142
        y_39 = input_142 = None
        shortcut_17 = torch.nn.functional.relu(y_40, inplace=False)
        y_40 = None
        input_143 = torch.conv2d(
            x_249,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_249 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_0_parameters_weight_ = (None)
        input_144 = torch.nn.functional.batch_norm(
            input_143,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_143 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_0_modules_1_parameters_bias_ = (None)
        input_145 = torch.nn.functional.relu(input_144, inplace=False)
        input_144 = None
        input_146 = torch.conv2d(
            input_145,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_145 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_0_parameters_weight_ = (None)
        input_147 = torch.nn.functional.batch_norm(
            input_146,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_146 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_1_modules_1_parameters_bias_ = (None)
        input_148 = torch.nn.functional.relu(input_147, inplace=False)
        input_147 = None
        input_149 = torch.conv2d(
            input_148,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_148 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_0_parameters_weight_ = (None)
        input_150 = torch.nn.functional.batch_norm(
            input_149,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_149 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_0_modules_2_modules_1_parameters_bias_ = (None)
        input_151 = torch.conv2d(
            x_263,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_263 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_0_parameters_weight_ = (None)
        input_152 = torch.nn.functional.batch_norm(
            input_151,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_151 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_0_modules_1_parameters_bias_ = (None)
        input_153 = torch.nn.functional.relu(input_152, inplace=False)
        input_152 = None
        input_154 = torch.conv2d(
            input_153,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_153 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_0_parameters_weight_ = (None)
        input_155 = torch.nn.functional.batch_norm(
            input_154,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_154 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_1_modules_1_modules_1_parameters_bias_ = (None)
        y_41 = input_150 + input_155
        input_150 = input_155 = None
        input_156 = torch.conv2d(
            x_277,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_277 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_0_parameters_weight_ = (None)
        input_157 = torch.nn.functional.batch_norm(
            input_156,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_156 = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_buffers_running_var_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_weight_ = l_self_modules_stage4_modules_1_modules_fuse_layers_modules_3_modules_2_modules_0_modules_1_parameters_bias_ = (None)
        y_42 = y_41 + input_157
        y_41 = input_157 = None
        y_43 = y_42 + x_291
        y_42 = x_291 = None
        shortcut_18 = torch.nn.functional.relu(y_43, inplace=False)
        y_43 = None
        x_292 = torch.conv2d(
            shortcut_15,
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
        x_293 = torch.nn.functional.batch_norm(
            x_292,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_292 = l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_weight_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn1_parameters_bias_ = (None)
        x_294 = torch.nn.functional.relu(x_293, inplace=True)
        x_293 = None
        x_295 = torch.conv2d(
            x_294,
            l_self_modules_incre_modules_modules_0_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_294 = l_self_modules_incre_modules_modules_0_modules_0_modules_conv2_parameters_weight_ = (None)
        x_296 = torch.nn.functional.batch_norm(
            x_295,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_295 = l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_weight_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn2_parameters_bias_ = (None)
        x_297 = torch.nn.functional.relu(x_296, inplace=True)
        x_296 = None
        x_298 = torch.conv2d(
            x_297,
            l_self_modules_incre_modules_modules_0_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_297 = l_self_modules_incre_modules_modules_0_modules_0_modules_conv3_parameters_weight_ = (None)
        x_299 = torch.nn.functional.batch_norm(
            x_298,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_298 = l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_weight_ = l_self_modules_incre_modules_modules_0_modules_0_modules_bn3_parameters_bias_ = (None)
        input_158 = torch.conv2d(
            shortcut_15,
            l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        shortcut_15 = l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_159 = torch.nn.functional.batch_norm(
            input_158,
            l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_158 = l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_incre_modules_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_299 += input_159
        x_300 = x_299
        x_299 = input_159 = None
        x_301 = torch.nn.functional.relu(x_300, inplace=True)
        x_300 = None
        x_302 = torch.conv2d(
            shortcut_16,
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
        x_303 = torch.nn.functional.batch_norm(
            x_302,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_302 = l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn1_parameters_bias_ = (None)
        x_304 = torch.nn.functional.relu(x_303, inplace=True)
        x_303 = None
        x_305 = torch.conv2d(
            x_304,
            l_self_modules_incre_modules_modules_1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_304 = l_self_modules_incre_modules_modules_1_modules_0_modules_conv2_parameters_weight_ = (None)
        x_306 = torch.nn.functional.batch_norm(
            x_305,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_305 = l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn2_parameters_bias_ = (None)
        x_307 = torch.nn.functional.relu(x_306, inplace=True)
        x_306 = None
        x_308 = torch.conv2d(
            x_307,
            l_self_modules_incre_modules_modules_1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_307 = l_self_modules_incre_modules_modules_1_modules_0_modules_conv3_parameters_weight_ = (None)
        x_309 = torch.nn.functional.batch_norm(
            x_308,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_308 = l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_weight_ = l_self_modules_incre_modules_modules_1_modules_0_modules_bn3_parameters_bias_ = (None)
        input_160 = torch.conv2d(
            shortcut_16,
            l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        shortcut_16 = l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_161 = torch.nn.functional.batch_norm(
            input_160,
            l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_160 = l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_incre_modules_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_309 += input_161
        x_310 = x_309
        x_309 = input_161 = None
        x_311 = torch.nn.functional.relu(x_310, inplace=True)
        x_310 = None
        input_162 = torch.conv2d(
            x_301,
            l_self_modules_downsamp_modules_modules_0_modules_0_parameters_weight_,
            l_self_modules_downsamp_modules_modules_0_modules_0_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_301 = (
            l_self_modules_downsamp_modules_modules_0_modules_0_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_0_modules_0_parameters_bias_ = None
        input_163 = torch.nn.functional.batch_norm(
            input_162,
            l_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_var_,
            l_self_modules_downsamp_modules_modules_0_modules_1_parameters_weight_,
            l_self_modules_downsamp_modules_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_162 = (
            l_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_downsamp_modules_modules_0_modules_1_buffers_running_var_
        ) = (
            l_self_modules_downsamp_modules_modules_0_modules_1_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_0_modules_1_parameters_bias_ = None
        input_164 = torch.nn.functional.relu(input_163, inplace=True)
        input_163 = None
        y_44 = x_311 + input_164
        x_311 = input_164 = None
        x_312 = torch.conv2d(
            shortcut_17,
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
        x_313 = torch.nn.functional.batch_norm(
            x_312,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_312 = l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn1_parameters_bias_ = (None)
        x_314 = torch.nn.functional.relu(x_313, inplace=True)
        x_313 = None
        x_315 = torch.conv2d(
            x_314,
            l_self_modules_incre_modules_modules_2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_314 = l_self_modules_incre_modules_modules_2_modules_0_modules_conv2_parameters_weight_ = (None)
        x_316 = torch.nn.functional.batch_norm(
            x_315,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_315 = l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn2_parameters_bias_ = (None)
        x_317 = torch.nn.functional.relu(x_316, inplace=True)
        x_316 = None
        x_318 = torch.conv2d(
            x_317,
            l_self_modules_incre_modules_modules_2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_317 = l_self_modules_incre_modules_modules_2_modules_0_modules_conv3_parameters_weight_ = (None)
        x_319 = torch.nn.functional.batch_norm(
            x_318,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_318 = l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_weight_ = l_self_modules_incre_modules_modules_2_modules_0_modules_bn3_parameters_bias_ = (None)
        input_165 = torch.conv2d(
            shortcut_17,
            l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        shortcut_17 = l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_166 = torch.nn.functional.batch_norm(
            input_165,
            l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_165 = l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_incre_modules_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_319 += input_166
        x_320 = x_319
        x_319 = input_166 = None
        x_321 = torch.nn.functional.relu(x_320, inplace=True)
        x_320 = None
        input_167 = torch.conv2d(
            y_44,
            l_self_modules_downsamp_modules_modules_1_modules_0_parameters_weight_,
            l_self_modules_downsamp_modules_modules_1_modules_0_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        y_44 = (
            l_self_modules_downsamp_modules_modules_1_modules_0_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_1_modules_0_parameters_bias_ = None
        input_168 = torch.nn.functional.batch_norm(
            input_167,
            l_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_var_,
            l_self_modules_downsamp_modules_modules_1_modules_1_parameters_weight_,
            l_self_modules_downsamp_modules_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_167 = (
            l_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_downsamp_modules_modules_1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_downsamp_modules_modules_1_modules_1_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_1_modules_1_parameters_bias_ = None
        input_169 = torch.nn.functional.relu(input_168, inplace=True)
        input_168 = None
        y_45 = x_321 + input_169
        x_321 = input_169 = None
        x_322 = torch.conv2d(
            shortcut_18,
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
        x_323 = torch.nn.functional.batch_norm(
            x_322,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_322 = l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn1_parameters_bias_ = (None)
        x_324 = torch.nn.functional.relu(x_323, inplace=True)
        x_323 = None
        x_325 = torch.conv2d(
            x_324,
            l_self_modules_incre_modules_modules_3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_324 = l_self_modules_incre_modules_modules_3_modules_0_modules_conv2_parameters_weight_ = (None)
        x_326 = torch.nn.functional.batch_norm(
            x_325,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_325 = l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn2_parameters_bias_ = (None)
        x_327 = torch.nn.functional.relu(x_326, inplace=True)
        x_326 = None
        x_328 = torch.conv2d(
            x_327,
            l_self_modules_incre_modules_modules_3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_327 = l_self_modules_incre_modules_modules_3_modules_0_modules_conv3_parameters_weight_ = (None)
        x_329 = torch.nn.functional.batch_norm(
            x_328,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_328 = l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_weight_ = l_self_modules_incre_modules_modules_3_modules_0_modules_bn3_parameters_bias_ = (None)
        input_170 = torch.conv2d(
            shortcut_18,
            l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        shortcut_18 = l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_171 = torch.nn.functional.batch_norm(
            input_170,
            l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_170 = l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_incre_modules_modules_3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_329 += input_171
        x_330 = x_329
        x_329 = input_171 = None
        x_331 = torch.nn.functional.relu(x_330, inplace=True)
        x_330 = None
        input_172 = torch.conv2d(
            y_45,
            l_self_modules_downsamp_modules_modules_2_modules_0_parameters_weight_,
            l_self_modules_downsamp_modules_modules_2_modules_0_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        y_45 = (
            l_self_modules_downsamp_modules_modules_2_modules_0_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_2_modules_0_parameters_bias_ = None
        input_173 = torch.nn.functional.batch_norm(
            input_172,
            l_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_var_,
            l_self_modules_downsamp_modules_modules_2_modules_1_parameters_weight_,
            l_self_modules_downsamp_modules_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_172 = (
            l_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_downsamp_modules_modules_2_modules_1_buffers_running_var_
        ) = (
            l_self_modules_downsamp_modules_modules_2_modules_1_parameters_weight_
        ) = l_self_modules_downsamp_modules_modules_2_modules_1_parameters_bias_ = None
        input_174 = torch.nn.functional.relu(input_173, inplace=True)
        input_173 = None
        y_46 = x_331 + input_174
        x_331 = input_174 = None
        input_175 = torch.conv2d(
            y_46,
            l_self_modules_final_layer_modules_0_parameters_weight_,
            l_self_modules_final_layer_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        y_46 = (
            l_self_modules_final_layer_modules_0_parameters_weight_
        ) = l_self_modules_final_layer_modules_0_parameters_bias_ = None
        input_176 = torch.nn.functional.batch_norm(
            input_175,
            l_self_modules_final_layer_modules_1_buffers_running_mean_,
            l_self_modules_final_layer_modules_1_buffers_running_var_,
            l_self_modules_final_layer_modules_1_parameters_weight_,
            l_self_modules_final_layer_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_175 = (
            l_self_modules_final_layer_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_final_layer_modules_1_buffers_running_var_
        ) = (
            l_self_modules_final_layer_modules_1_parameters_weight_
        ) = l_self_modules_final_layer_modules_1_parameters_bias_ = None
        input_177 = torch.nn.functional.relu(input_176, inplace=True)
        input_176 = None
        x_332 = torch.nn.functional.adaptive_avg_pool2d(input_177, 1)
        input_177 = None
        x_333 = x_332.flatten(1, -1)
        x_332 = None
        x_334 = torch.nn.functional.dropout(x_333, 0.0, False, False)
        x_333 = None
        x_335 = torch._C._nn.linear(
            x_334,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_334 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_335,)
