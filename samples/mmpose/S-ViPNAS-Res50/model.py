import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_conv_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_conv_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_final_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_final_layer_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_conv1_parameters_weight_ = (
            L_self_modules_backbone_modules_conv1_parameters_weight_
        )
        l_self_modules_backbone_modules_bn1_buffers_running_mean_ = (
            L_self_modules_backbone_modules_bn1_buffers_running_mean_
        )
        l_self_modules_backbone_modules_bn1_buffers_running_var_ = (
            L_self_modules_backbone_modules_bn1_buffers_running_var_
        )
        l_self_modules_backbone_modules_bn1_parameters_weight_ = (
            L_self_modules_backbone_modules_bn1_parameters_weight_
        )
        l_self_modules_backbone_modules_bn1_parameters_bias_ = (
            L_self_modules_backbone_modules_bn1_parameters_bias_
        )
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_3_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_3_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_3_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_3_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_3_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_3_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_3_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_conv_mask_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_conv_mask_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_conv_mask_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_conv_mask_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_
        l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_
        )
        l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_
        )
        l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_
        )
        l_self_modules_head_modules_final_layer_parameters_weight_ = (
            L_self_modules_head_modules_final_layer_parameters_weight_
        )
        l_self_modules_head_modules_final_layer_parameters_bias_ = (
            L_self_modules_head_modules_final_layer_parameters_bias_
        )
        x = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_conv1_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_backbone_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_backbone_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_bn1_parameters_weight_
        ) = l_self_modules_backbone_modules_bn1_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.nn.functional.max_pool2d(
            x_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_2 = None
        out = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_1 = torch.nn.functional.batch_norm(
            out,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_ = (None)
        out_2 = torch.nn.functional.relu(out_1, inplace=True)
        out_1 = None
        out_3 = torch.conv2d(
            out_2,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            16,
        )
        out_2 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_ = (None)
        out_4 = torch.nn.functional.batch_norm(
            out_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_3 = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_ = (None)
        out_5 = torch.nn.functional.relu(out_4, inplace=True)
        out_4 = None
        out_6 = torch.conv2d(
            out_5,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_5 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_ = (None)
        out_7 = torch.nn.functional.batch_norm(
            out_6,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_6 = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_ = (None)
        input_x = out_7.view(1, 80, 3072)
        input_x_1 = input_x.unsqueeze(1)
        input_x = None
        context_mask = torch.conv2d(
            out_7,
            l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_1 = context_mask.view(1, 1, 3072)
        context_mask = None
        context_mask_2 = torch.nn.functional.softmax(context_mask_1, 2, _stacklevel=5)
        context_mask_1 = None
        context_mask_3 = context_mask_2.unsqueeze(-1)
        context_mask_2 = None
        context = torch.matmul(input_x_1, context_mask_3)
        input_x_1 = context_mask_3 = None
        context_1 = context.view(1, 80, 1, 1)
        context = None
        input_1 = torch.conv2d(
            context_1,
            l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_1 = l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_2 = torch.nn.functional.layer_norm(
            input_1,
            (16, 1, 1),
            l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_1 = l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_8 = out_7 + input_4
        out_7 = input_4 = None
        input_5 = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_8 += input_6
        out_9 = out_8
        out_8 = input_6 = None
        out_10 = torch.nn.functional.relu(out_9, inplace=True)
        out_9 = None
        out_11 = torch.conv2d(
            out_10,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_12 = torch.nn.functional.batch_norm(
            out_11,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_11 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_ = (None)
        out_13 = torch.nn.functional.relu(out_12, inplace=True)
        out_12 = None
        out_14 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            16,
        )
        out_13 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_ = (None)
        out_15 = torch.nn.functional.batch_norm(
            out_14,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_14 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_ = (None)
        out_16 = torch.nn.functional.relu(out_15, inplace=True)
        out_15 = None
        out_17 = torch.conv2d(
            out_16,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_16 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_ = (None)
        out_18 = torch.nn.functional.batch_norm(
            out_17,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_17 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_ = (None)
        input_x_2 = out_18.view(1, 80, 3072)
        input_x_3 = input_x_2.unsqueeze(1)
        input_x_2 = None
        context_mask_4 = torch.conv2d(
            out_18,
            l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_5 = context_mask_4.view(1, 1, 3072)
        context_mask_4 = None
        context_mask_6 = torch.nn.functional.softmax(context_mask_5, 2, _stacklevel=5)
        context_mask_5 = None
        context_mask_7 = context_mask_6.unsqueeze(-1)
        context_mask_6 = None
        context_2 = torch.matmul(input_x_3, context_mask_7)
        input_x_3 = context_mask_7 = None
        context_3 = context_2.view(1, 80, 1, 1)
        context_2 = None
        input_7 = torch.conv2d(
            context_3,
            l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_3 = l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_8 = torch.nn.functional.layer_norm(
            input_7,
            (16, 1, 1),
            l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_7 = l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_9 = torch.nn.functional.relu(input_8, inplace=True)
        input_8 = None
        input_10 = torch.conv2d(
            input_9,
            l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_9 = l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_19 = out_18 + input_10
        out_18 = input_10 = None
        out_19 += out_10
        out_20 = out_19
        out_19 = out_10 = None
        out_21 = torch.nn.functional.relu(out_20, inplace=True)
        out_20 = None
        out_22 = torch.conv2d(
            out_21,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_23 = torch.nn.functional.batch_norm(
            out_22,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_22 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_ = (None)
        out_24 = torch.nn.functional.relu(out_23, inplace=True)
        out_23 = None
        out_25 = torch.conv2d(
            out_24,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            16,
        )
        out_24 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_ = (None)
        out_26 = torch.nn.functional.batch_norm(
            out_25,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_25 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_ = (None)
        out_27 = torch.nn.functional.relu(out_26, inplace=True)
        out_26 = None
        out_28 = torch.conv2d(
            out_27,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_27 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_ = (None)
        out_29 = torch.nn.functional.batch_norm(
            out_28,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_28 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_ = (None)
        input_x_4 = out_29.view(1, 80, 3072)
        input_x_5 = input_x_4.unsqueeze(1)
        input_x_4 = None
        context_mask_8 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_9 = context_mask_8.view(1, 1, 3072)
        context_mask_8 = None
        context_mask_10 = torch.nn.functional.softmax(context_mask_9, 2, _stacklevel=5)
        context_mask_9 = None
        context_mask_11 = context_mask_10.unsqueeze(-1)
        context_mask_10 = None
        context_4 = torch.matmul(input_x_5, context_mask_11)
        input_x_5 = context_mask_11 = None
        context_5 = context_4.view(1, 80, 1, 1)
        context_4 = None
        input_11 = torch.conv2d(
            context_5,
            l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_5 = l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_12 = torch.nn.functional.layer_norm(
            input_11,
            (16, 1, 1),
            l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_11 = l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_13 = torch.nn.functional.relu(input_12, inplace=True)
        input_12 = None
        input_14 = torch.conv2d(
            input_13,
            l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_13 = l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_30 = out_29 + input_14
        out_29 = input_14 = None
        out_30 += out_21
        out_31 = out_30
        out_30 = out_21 = None
        out_32 = torch.nn.functional.relu(out_31, inplace=True)
        out_31 = None
        out_33 = torch.conv2d(
            out_32,
            l_self_modules_backbone_modules_layer1_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_3_modules_conv1_parameters_weight_ = (
            None
        )
        out_34 = torch.nn.functional.batch_norm(
            out_33,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_33 = l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_3_modules_bn1_parameters_bias_ = (None)
        out_35 = torch.nn.functional.relu(out_34, inplace=True)
        out_34 = None
        out_36 = torch.conv2d(
            out_35,
            l_self_modules_backbone_modules_layer1_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            16,
        )
        out_35 = l_self_modules_backbone_modules_layer1_modules_3_modules_conv2_parameters_weight_ = (None)
        out_37 = torch.nn.functional.batch_norm(
            out_36,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_36 = l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_3_modules_bn2_parameters_bias_ = (None)
        out_38 = torch.nn.functional.relu(out_37, inplace=True)
        out_37 = None
        out_39 = torch.conv2d(
            out_38,
            l_self_modules_backbone_modules_layer1_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_38 = l_self_modules_backbone_modules_layer1_modules_3_modules_conv3_parameters_weight_ = (None)
        out_40 = torch.nn.functional.batch_norm(
            out_39,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_39 = l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_3_modules_bn3_parameters_bias_ = (None)
        input_x_6 = out_40.view(1, 80, 3072)
        input_x_7 = input_x_6.unsqueeze(1)
        input_x_6 = None
        context_mask_12 = torch.conv2d(
            out_40,
            l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_13 = context_mask_12.view(1, 1, 3072)
        context_mask_12 = None
        context_mask_14 = torch.nn.functional.softmax(context_mask_13, 2, _stacklevel=5)
        context_mask_13 = None
        context_mask_15 = context_mask_14.unsqueeze(-1)
        context_mask_14 = None
        context_6 = torch.matmul(input_x_7, context_mask_15)
        input_x_7 = context_mask_15 = None
        context_7 = context_6.view(1, 80, 1, 1)
        context_6 = None
        input_15 = torch.conv2d(
            context_7,
            l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_7 = l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_16 = torch.nn.functional.layer_norm(
            input_15,
            (16, 1, 1),
            l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_15 = l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_17 = torch.nn.functional.relu(input_16, inplace=True)
        input_16 = None
        input_18 = torch.conv2d(
            input_17,
            l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_17 = l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_41 = out_40 + input_18
        out_40 = input_18 = None
        out_41 += out_32
        out_42 = out_41
        out_41 = out_32 = None
        out_43 = torch.nn.functional.relu(out_42, inplace=True)
        out_42 = None
        out_44 = torch.conv2d(
            out_43,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_45 = torch.nn.functional.batch_norm(
            out_44,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_44 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_ = (None)
        out_46 = torch.nn.functional.relu(out_45, inplace=True)
        out_45 = None
        out_47 = torch.conv2d(
            out_46,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            16,
        )
        out_46 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_ = (None)
        out_48 = torch.nn.functional.batch_norm(
            out_47,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_47 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_ = (None)
        out_49 = torch.nn.functional.relu(out_48, inplace=True)
        out_48 = None
        out_50 = torch.conv2d(
            out_49,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_49 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_ = (None)
        out_51 = torch.nn.functional.batch_norm(
            out_50,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_50 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_ = (None)
        input_19 = torch.conv2d(
            out_43,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_43 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_20 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_19 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_51 += input_20
        out_52 = out_51
        out_51 = input_20 = None
        out_53 = torch.nn.functional.relu(out_52, inplace=True)
        out_52 = None
        out_54 = torch.conv2d(
            out_53,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_55 = torch.nn.functional.batch_norm(
            out_54,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_54 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_ = (None)
        out_56 = torch.nn.functional.relu(out_55, inplace=True)
        out_55 = None
        out_57 = torch.conv2d(
            out_56,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_56 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_ = (None)
        out_58 = torch.nn.functional.batch_norm(
            out_57,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_57 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_ = (None)
        out_59 = torch.nn.functional.relu(out_58, inplace=True)
        out_58 = None
        out_60 = torch.conv2d(
            out_59,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_59 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_ = (None)
        out_61 = torch.nn.functional.batch_norm(
            out_60,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_60 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_ = (None)
        out_61 += out_53
        out_62 = out_61
        out_61 = out_53 = None
        out_63 = torch.nn.functional.relu(out_62, inplace=True)
        out_62 = None
        out_64 = torch.conv2d(
            out_63,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_65 = torch.nn.functional.batch_norm(
            out_64,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_64 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_ = (None)
        out_66 = torch.nn.functional.relu(out_65, inplace=True)
        out_65 = None
        out_67 = torch.conv2d(
            out_66,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_66 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_ = (None)
        out_68 = torch.nn.functional.batch_norm(
            out_67,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_67 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_ = (None)
        out_69 = torch.nn.functional.relu(out_68, inplace=True)
        out_68 = None
        out_70 = torch.conv2d(
            out_69,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_69 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_ = (None)
        out_71 = torch.nn.functional.batch_norm(
            out_70,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_70 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_ = (None)
        out_71 += out_63
        out_72 = out_71
        out_71 = out_63 = None
        out_73 = torch.nn.functional.relu(out_72, inplace=True)
        out_72 = None
        out_74 = torch.conv2d(
            out_73,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_ = (
            None
        )
        out_75 = torch.nn.functional.batch_norm(
            out_74,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_74 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_ = (None)
        out_76 = torch.nn.functional.relu(out_75, inplace=True)
        out_75 = None
        out_77 = torch.conv2d(
            out_76,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_76 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_ = (None)
        out_78 = torch.nn.functional.batch_norm(
            out_77,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_77 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_ = (None)
        out_79 = torch.nn.functional.relu(out_78, inplace=True)
        out_78 = None
        out_80 = torch.conv2d(
            out_79,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_79 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_ = (None)
        out_81 = torch.nn.functional.batch_norm(
            out_80,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_80 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_ = (None)
        out_81 += out_73
        out_82 = out_81
        out_81 = out_73 = None
        out_83 = torch.nn.functional.relu(out_82, inplace=True)
        out_82 = None
        out_84 = torch.conv2d(
            out_83,
            l_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_ = (
            None
        )
        out_85 = torch.nn.functional.batch_norm(
            out_84,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_84 = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_ = (None)
        out_86 = torch.nn.functional.relu(out_85, inplace=True)
        out_85 = None
        out_87 = torch.conv2d(
            out_86,
            l_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_86 = l_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_ = (None)
        out_88 = torch.nn.functional.batch_norm(
            out_87,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_87 = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_ = (None)
        out_89 = torch.nn.functional.relu(out_88, inplace=True)
        out_88 = None
        out_90 = torch.conv2d(
            out_89,
            l_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_89 = l_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_ = (None)
        out_91 = torch.nn.functional.batch_norm(
            out_90,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_90 = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_ = (None)
        out_91 += out_83
        out_92 = out_91
        out_91 = out_83 = None
        out_93 = torch.nn.functional.relu(out_92, inplace=True)
        out_92 = None
        out_94 = torch.conv2d(
            out_93,
            l_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_ = (
            None
        )
        out_95 = torch.nn.functional.batch_norm(
            out_94,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_94 = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_ = (None)
        out_96 = torch.nn.functional.relu(out_95, inplace=True)
        out_95 = None
        out_97 = torch.conv2d(
            out_96,
            l_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_96 = l_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_ = (None)
        out_98 = torch.nn.functional.batch_norm(
            out_97,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_97 = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_ = (None)
        out_99 = torch.nn.functional.relu(out_98, inplace=True)
        out_98 = None
        out_100 = torch.conv2d(
            out_99,
            l_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_99 = l_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_ = (None)
        out_101 = torch.nn.functional.batch_norm(
            out_100,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_100 = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_ = (None)
        out_101 += out_93
        out_102 = out_101
        out_101 = out_93 = None
        out_103 = torch.nn.functional.relu(out_102, inplace=True)
        out_102 = None
        out_104 = torch.conv2d(
            out_103,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_105 = torch.nn.functional.batch_norm(
            out_104,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_104 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_ = (None)
        out_106 = torch.nn.functional.relu(out_105, inplace=True)
        out_105 = None
        out_107 = torch.conv2d(
            out_106,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            16,
        )
        out_106 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_ = (None)
        out_108 = torch.nn.functional.batch_norm(
            out_107,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_107 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_ = (None)
        out_109 = torch.nn.functional.relu(out_108, inplace=True)
        out_108 = None
        out_110 = torch.conv2d(
            out_109,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_109 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = (None)
        out_111 = torch.nn.functional.batch_norm(
            out_110,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_110 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = (None)
        input_x_8 = out_111.view(1, 304, 192)
        input_x_9 = input_x_8.unsqueeze(1)
        input_x_8 = None
        context_mask_16 = torch.conv2d(
            out_111,
            l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_17 = context_mask_16.view(1, 1, 192)
        context_mask_16 = None
        context_mask_18 = torch.nn.functional.softmax(context_mask_17, 2, _stacklevel=5)
        context_mask_17 = None
        context_mask_19 = context_mask_18.unsqueeze(-1)
        context_mask_18 = None
        context_8 = torch.matmul(input_x_9, context_mask_19)
        input_x_9 = context_mask_19 = None
        context_9 = context_8.view(1, 304, 1, 1)
        context_8 = None
        input_21 = torch.conv2d(
            context_9,
            l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_9 = l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_22 = torch.nn.functional.layer_norm(
            input_21,
            (19, 1, 1),
            l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_21 = l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_23 = torch.nn.functional.relu(input_22, inplace=True)
        input_22 = None
        input_24 = torch.conv2d(
            input_23,
            l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_23 = l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_112 = out_111 + input_24
        out_111 = input_24 = None
        input_25 = torch.conv2d(
            out_103,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_103 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_25 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_112 += input_26
        out_113 = out_112
        out_112 = input_26 = None
        out_114 = torch.nn.functional.relu(out_113, inplace=True)
        out_113 = None
        out_115 = torch.conv2d(
            out_114,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_116 = torch.nn.functional.batch_norm(
            out_115,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_115 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = (None)
        out_117 = torch.nn.functional.relu(out_116, inplace=True)
        out_116 = None
        out_118 = torch.conv2d(
            out_117,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_117 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_ = (None)
        out_119 = torch.nn.functional.batch_norm(
            out_118,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_118 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_ = (None)
        out_120 = torch.nn.functional.relu(out_119, inplace=True)
        out_119 = None
        out_121 = torch.conv2d(
            out_120,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_120 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_ = (None)
        out_122 = torch.nn.functional.batch_norm(
            out_121,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_121 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_ = (None)
        input_x_10 = out_122.view(1, 304, 192)
        input_x_11 = input_x_10.unsqueeze(1)
        input_x_10 = None
        context_mask_20 = torch.conv2d(
            out_122,
            l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_21 = context_mask_20.view(1, 1, 192)
        context_mask_20 = None
        context_mask_22 = torch.nn.functional.softmax(context_mask_21, 2, _stacklevel=5)
        context_mask_21 = None
        context_mask_23 = context_mask_22.unsqueeze(-1)
        context_mask_22 = None
        context_10 = torch.matmul(input_x_11, context_mask_23)
        input_x_11 = context_mask_23 = None
        context_11 = context_10.view(1, 304, 1, 1)
        context_10 = None
        input_27 = torch.conv2d(
            context_11,
            l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_11 = l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_28 = torch.nn.functional.layer_norm(
            input_27,
            (19, 1, 1),
            l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_27 = l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_29 = torch.nn.functional.relu(input_28, inplace=True)
        input_28 = None
        input_30 = torch.conv2d(
            input_29,
            l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_29 = l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_123 = out_122 + input_30
        out_122 = input_30 = None
        out_123 += out_114
        out_124 = out_123
        out_123 = out_114 = None
        out_125 = torch.nn.functional.relu(out_124, inplace=True)
        out_124 = None
        out_126 = torch.conv2d(
            out_125,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_127 = torch.nn.functional.batch_norm(
            out_126,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_126 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_ = (None)
        out_128 = torch.nn.functional.relu(out_127, inplace=True)
        out_127 = None
        out_129 = torch.conv2d(
            out_128,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_128 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_ = (None)
        out_130 = torch.nn.functional.batch_norm(
            out_129,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_129 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_ = (None)
        out_131 = torch.nn.functional.relu(out_130, inplace=True)
        out_130 = None
        out_132 = torch.conv2d(
            out_131,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_131 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_ = (None)
        out_133 = torch.nn.functional.batch_norm(
            out_132,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_132 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_ = (None)
        input_x_12 = out_133.view(1, 304, 192)
        input_x_13 = input_x_12.unsqueeze(1)
        input_x_12 = None
        context_mask_24 = torch.conv2d(
            out_133,
            l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_25 = context_mask_24.view(1, 1, 192)
        context_mask_24 = None
        context_mask_26 = torch.nn.functional.softmax(context_mask_25, 2, _stacklevel=5)
        context_mask_25 = None
        context_mask_27 = context_mask_26.unsqueeze(-1)
        context_mask_26 = None
        context_12 = torch.matmul(input_x_13, context_mask_27)
        input_x_13 = context_mask_27 = None
        context_13 = context_12.view(1, 304, 1, 1)
        context_12 = None
        input_31 = torch.conv2d(
            context_13,
            l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_13 = l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_32 = torch.nn.functional.layer_norm(
            input_31,
            (19, 1, 1),
            l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_31 = l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_33 = torch.nn.functional.relu(input_32, inplace=True)
        input_32 = None
        input_34 = torch.conv2d(
            input_33,
            l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_33 = l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_134 = out_133 + input_34
        out_133 = input_34 = None
        out_134 += out_125
        out_135 = out_134
        out_134 = out_125 = None
        out_136 = torch.nn.functional.relu(out_135, inplace=True)
        out_135 = None
        out_137 = torch.conv2d(
            out_136,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_ = (
            None
        )
        out_138 = torch.nn.functional.batch_norm(
            out_137,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_137 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_ = (None)
        out_139 = torch.nn.functional.relu(out_138, inplace=True)
        out_138 = None
        out_140 = torch.conv2d(
            out_139,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_139 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_ = (None)
        out_141 = torch.nn.functional.batch_norm(
            out_140,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_140 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_ = (None)
        out_142 = torch.nn.functional.relu(out_141, inplace=True)
        out_141 = None
        out_143 = torch.conv2d(
            out_142,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_142 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_ = (None)
        out_144 = torch.nn.functional.batch_norm(
            out_143,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_143 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_ = (None)
        input_x_14 = out_144.view(1, 304, 192)
        input_x_15 = input_x_14.unsqueeze(1)
        input_x_14 = None
        context_mask_28 = torch.conv2d(
            out_144,
            l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_29 = context_mask_28.view(1, 1, 192)
        context_mask_28 = None
        context_mask_30 = torch.nn.functional.softmax(context_mask_29, 2, _stacklevel=5)
        context_mask_29 = None
        context_mask_31 = context_mask_30.unsqueeze(-1)
        context_mask_30 = None
        context_14 = torch.matmul(input_x_15, context_mask_31)
        input_x_15 = context_mask_31 = None
        context_15 = context_14.view(1, 304, 1, 1)
        context_14 = None
        input_35 = torch.conv2d(
            context_15,
            l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_15 = l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_36 = torch.nn.functional.layer_norm(
            input_35,
            (19, 1, 1),
            l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_35 = l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_37 = torch.nn.functional.relu(input_36, inplace=True)
        input_36 = None
        input_38 = torch.conv2d(
            input_37,
            l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_37 = l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_145 = out_144 + input_38
        out_144 = input_38 = None
        out_145 += out_136
        out_146 = out_145
        out_145 = out_136 = None
        out_147 = torch.nn.functional.relu(out_146, inplace=True)
        out_146 = None
        out_148 = torch.conv2d(
            out_147,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_ = (
            None
        )
        out_149 = torch.nn.functional.batch_norm(
            out_148,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_148 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_ = (None)
        out_150 = torch.nn.functional.relu(out_149, inplace=True)
        out_149 = None
        out_151 = torch.conv2d(
            out_150,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_150 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_ = (None)
        out_152 = torch.nn.functional.batch_norm(
            out_151,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_151 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_ = (None)
        out_153 = torch.nn.functional.relu(out_152, inplace=True)
        out_152 = None
        out_154 = torch.conv2d(
            out_153,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_153 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_ = (None)
        out_155 = torch.nn.functional.batch_norm(
            out_154,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_154 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_ = (None)
        input_x_16 = out_155.view(1, 304, 192)
        input_x_17 = input_x_16.unsqueeze(1)
        input_x_16 = None
        context_mask_32 = torch.conv2d(
            out_155,
            l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_33 = context_mask_32.view(1, 1, 192)
        context_mask_32 = None
        context_mask_34 = torch.nn.functional.softmax(context_mask_33, 2, _stacklevel=5)
        context_mask_33 = None
        context_mask_35 = context_mask_34.unsqueeze(-1)
        context_mask_34 = None
        context_16 = torch.matmul(input_x_17, context_mask_35)
        input_x_17 = context_mask_35 = None
        context_17 = context_16.view(1, 304, 1, 1)
        context_16 = None
        input_39 = torch.conv2d(
            context_17,
            l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_17 = l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_40 = torch.nn.functional.layer_norm(
            input_39,
            (19, 1, 1),
            l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_39 = l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_41 = torch.nn.functional.relu(input_40, inplace=True)
        input_40 = None
        input_42 = torch.conv2d(
            input_41,
            l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_41 = l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_156 = out_155 + input_42
        out_155 = input_42 = None
        out_156 += out_147
        out_157 = out_156
        out_156 = out_147 = None
        out_158 = torch.nn.functional.relu(out_157, inplace=True)
        out_157 = None
        out_159 = torch.conv2d(
            out_158,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_ = (
            None
        )
        out_160 = torch.nn.functional.batch_norm(
            out_159,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_159 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_ = (None)
        out_161 = torch.nn.functional.relu(out_160, inplace=True)
        out_160 = None
        out_162 = torch.conv2d(
            out_161,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_161 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_ = (None)
        out_163 = torch.nn.functional.batch_norm(
            out_162,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_162 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_ = (None)
        out_164 = torch.nn.functional.relu(out_163, inplace=True)
        out_163 = None
        out_165 = torch.conv2d(
            out_164,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_164 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_ = (None)
        out_166 = torch.nn.functional.batch_norm(
            out_165,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_165 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_ = (None)
        input_x_18 = out_166.view(1, 304, 192)
        input_x_19 = input_x_18.unsqueeze(1)
        input_x_18 = None
        context_mask_36 = torch.conv2d(
            out_166,
            l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_37 = context_mask_36.view(1, 1, 192)
        context_mask_36 = None
        context_mask_38 = torch.nn.functional.softmax(context_mask_37, 2, _stacklevel=5)
        context_mask_37 = None
        context_mask_39 = context_mask_38.unsqueeze(-1)
        context_mask_38 = None
        context_18 = torch.matmul(input_x_19, context_mask_39)
        input_x_19 = context_mask_39 = None
        context_19 = context_18.view(1, 304, 1, 1)
        context_18 = None
        input_43 = torch.conv2d(
            context_19,
            l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_19 = l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_44 = torch.nn.functional.layer_norm(
            input_43,
            (19, 1, 1),
            l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_43 = l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.relu(input_44, inplace=True)
        input_44 = None
        input_46 = torch.conv2d(
            input_45,
            l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_45 = l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_167 = out_166 + input_46
        out_166 = input_46 = None
        out_167 += out_158
        out_168 = out_167
        out_167 = out_158 = None
        out_169 = torch.nn.functional.relu(out_168, inplace=True)
        out_168 = None
        out_170 = torch.conv2d(
            out_169,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_ = (
            None
        )
        out_171 = torch.nn.functional.batch_norm(
            out_170,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_170 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_ = (None)
        out_172 = torch.nn.functional.relu(out_171, inplace=True)
        out_171 = None
        out_173 = torch.conv2d(
            out_172,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_172 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_ = (None)
        out_174 = torch.nn.functional.batch_norm(
            out_173,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_173 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_ = (None)
        out_175 = torch.nn.functional.relu(out_174, inplace=True)
        out_174 = None
        out_176 = torch.conv2d(
            out_175,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_175 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_ = (None)
        out_177 = torch.nn.functional.batch_norm(
            out_176,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_176 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_ = (None)
        input_x_20 = out_177.view(1, 304, 192)
        input_x_21 = input_x_20.unsqueeze(1)
        input_x_20 = None
        context_mask_40 = torch.conv2d(
            out_177,
            l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_41 = context_mask_40.view(1, 1, 192)
        context_mask_40 = None
        context_mask_42 = torch.nn.functional.softmax(context_mask_41, 2, _stacklevel=5)
        context_mask_41 = None
        context_mask_43 = context_mask_42.unsqueeze(-1)
        context_mask_42 = None
        context_20 = torch.matmul(input_x_21, context_mask_43)
        input_x_21 = context_mask_43 = None
        context_21 = context_20.view(1, 304, 1, 1)
        context_20 = None
        input_47 = torch.conv2d(
            context_21,
            l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_21 = l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_48 = torch.nn.functional.layer_norm(
            input_47,
            (19, 1, 1),
            l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_47 = l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_49 = torch.nn.functional.relu(input_48, inplace=True)
        input_48 = None
        input_50 = torch.conv2d(
            input_49,
            l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_49 = l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_178 = out_177 + input_50
        out_177 = input_50 = None
        out_178 += out_169
        out_179 = out_178
        out_178 = out_169 = None
        out_180 = torch.nn.functional.relu(out_179, inplace=True)
        out_179 = None
        out_181 = torch.conv2d(
            out_180,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_182 = torch.nn.functional.batch_norm(
            out_181,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_181 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (None)
        out_183 = torch.nn.functional.relu(out_182, inplace=True)
        out_182 = None
        out_184 = torch.conv2d(
            out_183,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            16,
        )
        out_183 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_ = (None)
        out_185 = torch.nn.functional.batch_norm(
            out_184,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_184 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_ = (None)
        out_186 = torch.nn.functional.relu(out_185, inplace=True)
        out_185 = None
        out_187 = torch.conv2d(
            out_186,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_186 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = (None)
        out_188 = torch.nn.functional.batch_norm(
            out_187,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_187 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = (None)
        input_x_22 = out_188.view(1, 608, 48)
        input_x_23 = input_x_22.unsqueeze(1)
        input_x_22 = None
        context_mask_44 = torch.conv2d(
            out_188,
            l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_45 = context_mask_44.view(1, 1, 48)
        context_mask_44 = None
        context_mask_46 = torch.nn.functional.softmax(context_mask_45, 2, _stacklevel=5)
        context_mask_45 = None
        context_mask_47 = context_mask_46.unsqueeze(-1)
        context_mask_46 = None
        context_22 = torch.matmul(input_x_23, context_mask_47)
        input_x_23 = context_mask_47 = None
        context_23 = context_22.view(1, 608, 1, 1)
        context_22 = None
        input_51 = torch.conv2d(
            context_23,
            l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_23 = l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_52 = torch.nn.functional.layer_norm(
            input_51,
            (38, 1, 1),
            l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_51 = l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_53 = torch.nn.functional.relu(input_52, inplace=True)
        input_52 = None
        input_54 = torch.conv2d(
            input_53,
            l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_53 = l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_189 = out_188 + input_54
        out_188 = input_54 = None
        input_55 = torch.conv2d(
            out_180,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_180 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_55 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_189 += input_56
        out_190 = out_189
        out_189 = input_56 = None
        out_191 = torch.nn.functional.relu(out_190, inplace=True)
        out_190 = None
        out_192 = torch.conv2d(
            out_191,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_193 = torch.nn.functional.batch_norm(
            out_192,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_192 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (None)
        out_194 = torch.nn.functional.relu(out_193, inplace=True)
        out_193 = None
        out_195 = torch.conv2d(
            out_194,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_194 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_ = (None)
        out_196 = torch.nn.functional.batch_norm(
            out_195,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_195 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_ = (None)
        out_197 = torch.nn.functional.relu(out_196, inplace=True)
        out_196 = None
        out_198 = torch.conv2d(
            out_197,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_197 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_ = (None)
        out_199 = torch.nn.functional.batch_norm(
            out_198,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_198 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_ = (None)
        input_x_24 = out_199.view(1, 608, 48)
        input_x_25 = input_x_24.unsqueeze(1)
        input_x_24 = None
        context_mask_48 = torch.conv2d(
            out_199,
            l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_49 = context_mask_48.view(1, 1, 48)
        context_mask_48 = None
        context_mask_50 = torch.nn.functional.softmax(context_mask_49, 2, _stacklevel=5)
        context_mask_49 = None
        context_mask_51 = context_mask_50.unsqueeze(-1)
        context_mask_50 = None
        context_24 = torch.matmul(input_x_25, context_mask_51)
        input_x_25 = context_mask_51 = None
        context_25 = context_24.view(1, 608, 1, 1)
        context_24 = None
        input_57 = torch.conv2d(
            context_25,
            l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_25 = l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_58 = torch.nn.functional.layer_norm(
            input_57,
            (38, 1, 1),
            l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_57 = l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_59 = torch.nn.functional.relu(input_58, inplace=True)
        input_58 = None
        input_60 = torch.conv2d(
            input_59,
            l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_59 = l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_200 = out_199 + input_60
        out_199 = input_60 = None
        out_200 += out_191
        out_201 = out_200
        out_200 = out_191 = None
        out_202 = torch.nn.functional.relu(out_201, inplace=True)
        out_201 = None
        out_203 = torch.conv2d(
            out_202,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_204 = torch.nn.functional.batch_norm(
            out_203,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_203 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_ = (None)
        out_205 = torch.nn.functional.relu(out_204, inplace=True)
        out_204 = None
        out_206 = torch.conv2d(
            out_205,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            16,
        )
        out_205 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_ = (None)
        out_207 = torch.nn.functional.batch_norm(
            out_206,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_206 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_ = (None)
        out_208 = torch.nn.functional.relu(out_207, inplace=True)
        out_207 = None
        out_209 = torch.conv2d(
            out_208,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_208 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_ = (None)
        out_210 = torch.nn.functional.batch_norm(
            out_209,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_209 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_ = (None)
        input_x_26 = out_210.view(1, 608, 48)
        input_x_27 = input_x_26.unsqueeze(1)
        input_x_26 = None
        context_mask_52 = torch.conv2d(
            out_210,
            l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_conv_mask_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_conv_mask_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_conv_mask_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_conv_mask_parameters_bias_ = (None)
        context_mask_53 = context_mask_52.view(1, 1, 48)
        context_mask_52 = None
        context_mask_54 = torch.nn.functional.softmax(context_mask_53, 2, _stacklevel=5)
        context_mask_53 = None
        context_mask_55 = context_mask_54.unsqueeze(-1)
        context_mask_54 = None
        context_26 = torch.matmul(input_x_27, context_mask_55)
        input_x_27 = context_mask_55 = None
        context_27 = context_26.view(1, 608, 1, 1)
        context_26 = None
        input_61 = torch.conv2d(
            context_27,
            l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        context_27 = l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_0_parameters_bias_ = (None)
        input_62 = torch.nn.functional.layer_norm(
            input_61,
            (38, 1, 1),
            l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_,
            1e-05,
        )
        input_61 = l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_1_parameters_bias_ = (None)
        input_63 = torch.nn.functional.relu(input_62, inplace=True)
        input_62 = None
        input_64 = torch.conv2d(
            input_63,
            l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_63 = l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_attention_modules_channel_add_conv_modules_3_parameters_bias_ = (None)
        out_211 = out_210 + input_64
        out_210 = input_64 = None
        out_211 += out_202
        out_212 = out_211
        out_211 = out_202 = None
        out_213 = torch.nn.functional.relu(out_212, inplace=True)
        out_212 = None
        input_65 = torch.conv_transpose2d(
            out_213,
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            16,
            (1, 1),
        )
        out_213 = (
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        ) = None
        input_66 = torch.nn.functional.batch_norm(
            input_65,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_65 = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = None
        input_67 = torch.nn.functional.relu(input_66, inplace=True)
        input_66 = None
        input_68 = torch.conv_transpose2d(
            input_67,
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            16,
            (1, 1),
        )
        input_67 = (
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        ) = None
        input_69 = torch.nn.functional.batch_norm(
            input_68,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_68 = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = None
        input_70 = torch.nn.functional.relu(input_69, inplace=True)
        input_69 = None
        input_71 = torch.conv_transpose2d(
            input_70,
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            16,
            (1, 1),
        )
        input_70 = (
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        ) = None
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_71 = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = None
        input_73 = torch.nn.functional.relu(input_72, inplace=True)
        input_72 = None
        x_4 = torch.conv2d(
            input_73,
            l_self_modules_head_modules_final_layer_parameters_weight_,
            l_self_modules_head_modules_final_layer_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_73 = (
            l_self_modules_head_modules_final_layer_parameters_weight_
        ) = l_self_modules_head_modules_final_layer_parameters_bias_ = None
        return (x_4,)
