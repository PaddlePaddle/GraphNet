import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reduce_layers_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reduce_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsamples_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsamples_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_downsamples_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_downsamples_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsamples_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsamples_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsamples_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_downsamples_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_downsamples_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsamples_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_out_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_out_convs_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_out_convs_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_out_convs_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_out_convs_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_out_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_out_convs_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_out_convs_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_out_convs_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_out_convs_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_out_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_out_convs_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_out_convs_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_out_convs_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_out_convs_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_obj_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_obj_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_obj_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_obj_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_cls_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_cls_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_obj_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_obj_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_bbox_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_bbox_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_stem_modules_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage1_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage1_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stage1_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage4_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_reduce_layers_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_reduce_layers_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_parameters_weight_ = L_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_parameters_weight_
        l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_parameters_bias_ = L_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_parameters_bias_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_reduce_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_reduce_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_downsamples_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_downsamples_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_downsamples_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_downsamples_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_downsamples_modules_0_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_downsamples_modules_0_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_downsamples_modules_0_modules_bn_parameters_weight_ = L_self_modules_neck_modules_downsamples_modules_0_modules_bn_parameters_weight_
        l_self_modules_neck_modules_downsamples_modules_0_modules_bn_parameters_bias_ = L_self_modules_neck_modules_downsamples_modules_0_modules_bn_parameters_bias_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_downsamples_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_downsamples_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_downsamples_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_downsamples_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_downsamples_modules_1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_downsamples_modules_1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_downsamples_modules_1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_downsamples_modules_1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_downsamples_modules_1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_downsamples_modules_1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_conv_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_conv_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_parameters_weight_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_parameters_weight_
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_parameters_bias_ = L_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_out_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_out_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_out_convs_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_out_convs_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_out_convs_modules_0_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_out_convs_modules_0_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_out_convs_modules_0_modules_bn_parameters_weight_ = L_self_modules_neck_modules_out_convs_modules_0_modules_bn_parameters_weight_
        l_self_modules_neck_modules_out_convs_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_neck_modules_out_convs_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_neck_modules_out_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_out_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_out_convs_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_out_convs_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_out_convs_modules_1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_out_convs_modules_1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_out_convs_modules_1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_out_convs_modules_1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_out_convs_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_neck_modules_out_convs_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_neck_modules_out_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_out_convs_modules_2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_out_convs_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_out_convs_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_out_convs_modules_2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_out_convs_modules_2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_out_convs_modules_2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_out_convs_modules_2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_out_convs_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_neck_modules_out_convs_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_obj_modules_0_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_obj_modules_0_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_obj_modules_0_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_obj_modules_0_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_kpt_modules_0_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_kpt_modules_0_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_kpt_modules_0_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_kpt_modules_0_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_obj_modules_1_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_obj_modules_1_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_obj_modules_1_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_obj_modules_1_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_kpt_modules_1_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_kpt_modules_1_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_kpt_modules_1_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_kpt_modules_1_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_conv_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_conv_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_buffers_running_var_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_buffers_running_var_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_parameters_weight_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_parameters_weight_
        l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_parameters_bias_ = L_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_cls_modules_2_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_cls_modules_2_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_cls_modules_2_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_cls_modules_2_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_obj_modules_2_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_obj_modules_2_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_obj_modules_2_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_obj_modules_2_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_2_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_bbox_modules_2_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_2_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_bbox_modules_2_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_kpt_modules_2_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_kpt_modules_2_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_kpt_modules_2_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_kpt_modules_2_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_2_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_2_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_2_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_2_parameters_bias_
        patch_top_left = l_inputs_[
            (Ellipsis, slice(None, None, 2), slice(None, None, 2))
        ]
        patch_top_right = l_inputs_[(Ellipsis, slice(None, None, 2), slice(1, None, 2))]
        patch_bot_left = l_inputs_[(Ellipsis, slice(1, None, 2), slice(None, None, 2))]
        patch_bot_right = l_inputs_[(Ellipsis, slice(1, None, 2), slice(1, None, 2))]
        l_inputs_ = None
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1
        )
        patch_top_left = patch_bot_left = patch_top_right = patch_bot_right = None
        x_1 = torch.conv2d(
            x,
            l_self_modules_backbone_modules_stem_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x = l_self_modules_backbone_modules_stem_modules_conv_modules_conv_parameters_weight_ = (None)
        x_2 = torch.nn.functional.batch_norm(
            x_1,
            l_self_modules_backbone_modules_stem_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_1 = l_self_modules_backbone_modules_stem_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_conv_modules_bn_parameters_bias_ = (None)
        sigmoid = torch.sigmoid(x_2)
        x_3 = x_2 * sigmoid
        x_2 = sigmoid = None
        x_4 = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_stage1_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_backbone_modules_stage1_modules_0_modules_conv_parameters_weight_ = (None)
        x_5 = torch.nn.functional.batch_norm(
            x_4,
            l_self_modules_backbone_modules_stage1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_4 = l_self_modules_backbone_modules_stage1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage1_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_stage1_modules_0_modules_bn_parameters_bias_
        ) = None
        sigmoid_1 = torch.sigmoid(x_5)
        x_6 = x_5 * sigmoid_1
        x_5 = sigmoid_1 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_7 = l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage1_modules_1_modules_short_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_2 = torch.sigmoid(x_8)
        x_9 = x_8 * sigmoid_2
        x_8 = sigmoid_2 = None
        x_10 = torch.conv2d(
            x_6,
            l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_conv_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_10 = l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage1_modules_1_modules_main_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_3 = torch.sigmoid(x_11)
        x_12 = x_11 * sigmoid_3
        x_11 = sigmoid_3 = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_13 = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_4 = torch.sigmoid(x_14)
        x_15 = x_14 * sigmoid_4
        x_14 = sigmoid_4 = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_16 = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_5 = torch.sigmoid(x_17)
        x_18 = x_17 * sigmoid_5
        x_17 = sigmoid_5 = None
        input_1 = x_18 + x_12
        x_18 = x_12 = None
        x_final = torch.cat((input_1, x_9), dim=1)
        input_1 = x_9 = None
        x_19 = torch.conv2d(
            x_final,
            l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final = l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_19 = l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_6 = torch.sigmoid(x_20)
        x_21 = x_20 * sigmoid_6
        x_20 = sigmoid_6 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_backbone_modules_stage2_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_backbone_modules_stage2_modules_0_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_22 = l_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_bias_
        ) = None
        sigmoid_7 = torch.sigmoid(x_23)
        x_24 = x_23 * sigmoid_7
        x_23 = sigmoid_7 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_25 = l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_8 = torch.sigmoid(x_26)
        x_27 = x_26 * sigmoid_8
        x_26 = sigmoid_8 = None
        x_28 = torch.conv2d(
            x_24,
            l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_24 = l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_28 = l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_9 = torch.sigmoid(x_29)
        x_30 = x_29 * sigmoid_9
        x_29 = sigmoid_9 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_31 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_10 = torch.sigmoid(x_32)
        x_33 = x_32 * sigmoid_10
        x_32 = sigmoid_10 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_34 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_11 = torch.sigmoid(x_35)
        x_36 = x_35 * sigmoid_11
        x_35 = sigmoid_11 = None
        input_2 = x_36 + x_30
        x_36 = x_30 = None
        x_37 = torch.conv2d(
            input_2,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_37 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_12 = torch.sigmoid(x_38)
        x_39 = x_38 * sigmoid_12
        x_38 = sigmoid_12 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_40 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_13 = torch.sigmoid(x_41)
        x_42 = x_41 * sigmoid_13
        x_41 = sigmoid_13 = None
        input_3 = x_42 + input_2
        x_42 = input_2 = None
        x_43 = torch.conv2d(
            input_3,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_43 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_14 = torch.sigmoid(x_44)
        x_45 = x_44 * sigmoid_14
        x_44 = sigmoid_14 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_46 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_15 = torch.sigmoid(x_47)
        x_48 = x_47 * sigmoid_15
        x_47 = sigmoid_15 = None
        input_4 = x_48 + input_3
        x_48 = input_3 = None
        x_final_1 = torch.cat((input_4, x_27), dim=1)
        input_4 = x_27 = None
        x_49 = torch.conv2d(
            x_final_1,
            l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final_1 = l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_49 = l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_16 = torch.sigmoid(x_50)
        x_51 = x_50 * sigmoid_16
        x_50 = sigmoid_16 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_backbone_modules_stage3_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage3_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_52 = l_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_bias_
        ) = None
        sigmoid_17 = torch.sigmoid(x_53)
        x_54 = x_53 * sigmoid_17
        x_53 = sigmoid_17 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_55 = l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_18 = torch.sigmoid(x_56)
        x_57 = x_56 * sigmoid_18
        x_56 = sigmoid_18 = None
        x_58 = torch.conv2d(
            x_54,
            l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_58 = l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_19 = torch.sigmoid(x_59)
        x_60 = x_59 * sigmoid_19
        x_59 = sigmoid_19 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_61 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_20 = torch.sigmoid(x_62)
        x_63 = x_62 * sigmoid_20
        x_62 = sigmoid_20 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_64 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_21 = torch.sigmoid(x_65)
        x_66 = x_65 * sigmoid_21
        x_65 = sigmoid_21 = None
        input_5 = x_66 + x_60
        x_66 = x_60 = None
        x_67 = torch.conv2d(
            input_5,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_67 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_22 = torch.sigmoid(x_68)
        x_69 = x_68 * sigmoid_22
        x_68 = sigmoid_22 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_70 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_23 = torch.sigmoid(x_71)
        x_72 = x_71 * sigmoid_23
        x_71 = sigmoid_23 = None
        input_6 = x_72 + input_5
        x_72 = input_5 = None
        x_73 = torch.conv2d(
            input_6,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_73 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_24 = torch.sigmoid(x_74)
        x_75 = x_74 * sigmoid_24
        x_74 = sigmoid_24 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_76 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_25 = torch.sigmoid(x_77)
        x_78 = x_77 * sigmoid_25
        x_77 = sigmoid_25 = None
        input_7 = x_78 + input_6
        x_78 = input_6 = None
        x_final_2 = torch.cat((input_7, x_57), dim=1)
        input_7 = x_57 = None
        x_79 = torch.conv2d(
            x_final_2,
            l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final_2 = l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_79 = l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_26 = torch.sigmoid(x_80)
        x_81 = x_80 * sigmoid_26
        x_80 = sigmoid_26 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_backbone_modules_stage4_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage4_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_82 = l_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_bias_
        ) = None
        sigmoid_27 = torch.sigmoid(x_83)
        x_84 = x_83 * sigmoid_27
        x_83 = sigmoid_27 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_85 = l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_28 = torch.sigmoid(x_86)
        x_87 = x_86 * sigmoid_28
        x_86 = sigmoid_28 = None
        max_pool2d = torch.nn.functional.max_pool2d(
            x_87, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_1 = torch.nn.functional.max_pool2d(
            x_87, 9, 1, 4, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_2 = torch.nn.functional.max_pool2d(
            x_87, 13, 1, 6, 1, ceil_mode=False, return_indices=False
        )
        cat_4 = torch.cat([x_87, max_pool2d, max_pool2d_1, max_pool2d_2], dim=1)
        x_87 = max_pool2d = max_pool2d_1 = max_pool2d_2 = None
        x_88 = torch.conv2d(
            cat_4,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_4 = l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_88 = l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_29 = torch.sigmoid(x_89)
        x_90 = x_89 * sigmoid_29
        x_89 = sigmoid_29 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_91 = l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_30 = torch.sigmoid(x_92)
        x_93 = x_92 * sigmoid_30
        x_92 = sigmoid_30 = None
        x_94 = torch.conv2d(
            x_90,
            l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_94 = l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_31 = torch.sigmoid(x_95)
        x_96 = x_95 * sigmoid_31
        x_95 = sigmoid_31 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_97 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_32 = torch.sigmoid(x_98)
        x_99 = x_98 * sigmoid_32
        x_98 = sigmoid_32 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_100 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_33 = torch.sigmoid(x_101)
        x_102 = x_101 * sigmoid_33
        x_101 = sigmoid_33 = None
        x_final_3 = torch.cat((x_102, x_93), dim=1)
        x_102 = x_93 = None
        x_103 = torch.conv2d(
            x_final_3,
            l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final_3 = l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_103 = l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_34 = torch.sigmoid(x_104)
        x_105 = x_104 * sigmoid_34
        x_104 = sigmoid_34 = None
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_neck_modules_reduce_layers_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_105 = l_self_modules_neck_modules_reduce_layers_modules_0_modules_conv_parameters_weight_ = (None)
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_106 = l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_neck_modules_reduce_layers_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_35 = torch.sigmoid(x_107)
        x_108 = x_107 * sigmoid_35
        x_107 = sigmoid_35 = None
        upsample_feat = torch.nn.functional.interpolate(
            x_108, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        cat_6 = torch.cat([upsample_feat, x_81], 1)
        upsample_feat = x_81 = None
        x_109 = torch.conv2d(
            cat_6,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_109 = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_short_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_36 = torch.sigmoid(x_110)
        x_111 = x_110 * sigmoid_36
        x_110 = sigmoid_36 = None
        x_112 = torch.conv2d(
            cat_6,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_6 = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_conv_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_112 = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_main_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_37 = torch.sigmoid(x_113)
        x_114 = x_113 * sigmoid_37
        x_113 = sigmoid_37 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_114 = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_115 = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_38 = torch.sigmoid(x_116)
        x_117 = x_116 * sigmoid_38
        x_116 = sigmoid_38 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_118 = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_39 = torch.sigmoid(x_119)
        x_120 = x_119 * sigmoid_39
        x_119 = sigmoid_39 = None
        x_final_4 = torch.cat((x_120, x_111), dim=1)
        x_120 = x_111 = None
        x_121 = torch.conv2d(
            x_final_4,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final_4 = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_121 = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_top_down_blocks_modules_0_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_40 = torch.sigmoid(x_122)
        x_123 = x_122 * sigmoid_40
        x_122 = sigmoid_40 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_neck_modules_reduce_layers_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_123 = l_self_modules_neck_modules_reduce_layers_modules_1_modules_conv_parameters_weight_ = (None)
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_124 = l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_reduce_layers_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_41 = torch.sigmoid(x_125)
        x_126 = x_125 * sigmoid_41
        x_125 = sigmoid_41 = None
        upsample_feat_1 = torch.nn.functional.interpolate(
            x_126, None, 2.0, "nearest", None, recompute_scale_factor=None
        )
        cat_8 = torch.cat([upsample_feat_1, x_51], 1)
        upsample_feat_1 = x_51 = None
        x_127 = torch.conv2d(
            cat_8,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_127 = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_short_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_42 = torch.sigmoid(x_128)
        x_129 = x_128 * sigmoid_42
        x_128 = sigmoid_42 = None
        x_130 = torch.conv2d(
            cat_8,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_8 = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_conv_parameters_weight_ = (None)
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_130 = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_main_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_43 = torch.sigmoid(x_131)
        x_132 = x_131 * sigmoid_43
        x_131 = sigmoid_43 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_132 = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_133 = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_44 = torch.sigmoid(x_134)
        x_135 = x_134 * sigmoid_44
        x_134 = sigmoid_44 = None
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_135 = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_136 = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_45 = torch.sigmoid(x_137)
        x_138 = x_137 * sigmoid_45
        x_137 = sigmoid_45 = None
        x_final_5 = torch.cat((x_138, x_129), dim=1)
        x_138 = x_129 = None
        x_139 = torch.conv2d(
            x_final_5,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final_5 = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_139 = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_top_down_blocks_modules_1_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_46 = torch.sigmoid(x_140)
        x_141 = x_140 * sigmoid_46
        x_140 = sigmoid_46 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_neck_modules_downsamples_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_downsamples_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_neck_modules_downsamples_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_downsamples_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_downsamples_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_downsamples_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_142 = l_self_modules_neck_modules_downsamples_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_downsamples_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_downsamples_modules_0_modules_bn_parameters_weight_ = l_self_modules_neck_modules_downsamples_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_47 = torch.sigmoid(x_143)
        x_144 = x_143 * sigmoid_47
        x_143 = sigmoid_47 = None
        cat_10 = torch.cat([x_144, x_126], 1)
        x_144 = x_126 = None
        x_145 = torch.conv2d(
            cat_10,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_145 = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_short_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_48 = torch.sigmoid(x_146)
        x_147 = x_146 * sigmoid_48
        x_146 = sigmoid_48 = None
        x_148 = torch.conv2d(
            cat_10,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_10 = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_148 = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_main_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_49 = torch.sigmoid(x_149)
        x_150 = x_149 * sigmoid_49
        x_149 = sigmoid_49 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_150 = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_151 = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_50 = torch.sigmoid(x_152)
        x_153 = x_152 * sigmoid_50
        x_152 = sigmoid_50 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_154 = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_51 = torch.sigmoid(x_155)
        x_156 = x_155 * sigmoid_51
        x_155 = sigmoid_51 = None
        x_final_6 = torch.cat((x_156, x_147), dim=1)
        x_156 = x_147 = None
        x_157 = torch.conv2d(
            x_final_6,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final_6 = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_157 = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_bottom_up_blocks_modules_0_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_52 = torch.sigmoid(x_158)
        x_159 = x_158 * sigmoid_52
        x_158 = sigmoid_52 = None
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_neck_modules_downsamples_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_downsamples_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_neck_modules_downsamples_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_downsamples_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_downsamples_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_downsamples_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_160 = l_self_modules_neck_modules_downsamples_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_downsamples_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_downsamples_modules_1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_downsamples_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_53 = torch.sigmoid(x_161)
        x_162 = x_161 * sigmoid_53
        x_161 = sigmoid_53 = None
        cat_12 = torch.cat([x_162, x_108], 1)
        x_162 = x_108 = None
        x_163 = torch.conv2d(
            cat_12,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_163 = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_short_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_54 = torch.sigmoid(x_164)
        x_165 = x_164 * sigmoid_54
        x_164 = sigmoid_54 = None
        x_166 = torch.conv2d(
            cat_12,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_12 = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_conv_parameters_weight_ = (None)
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_166 = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_main_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_55 = torch.sigmoid(x_167)
        x_168 = x_167 * sigmoid_55
        x_167 = sigmoid_55 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_169 = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_56 = torch.sigmoid(x_170)
        x_171 = x_170 * sigmoid_56
        x_170 = sigmoid_56 = None
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_171 = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_172 = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_57 = torch.sigmoid(x_173)
        x_174 = x_173 * sigmoid_57
        x_173 = sigmoid_57 = None
        x_final_7 = torch.cat((x_174, x_165), dim=1)
        x_174 = x_165 = None
        x_175 = torch.conv2d(
            x_final_7,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final_7 = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_175 = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_neck_modules_bottom_up_blocks_modules_1_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_58 = torch.sigmoid(x_176)
        x_177 = x_176 * sigmoid_58
        x_176 = sigmoid_58 = None
        x_178 = torch.conv2d(
            x_141,
            l_self_modules_neck_modules_out_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_neck_modules_out_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_neck_modules_out_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_out_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_out_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_out_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_178 = l_self_modules_neck_modules_out_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_out_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_out_convs_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_neck_modules_out_convs_modules_0_modules_bn_parameters_bias_
        ) = None
        sigmoid_59 = torch.sigmoid(x_179)
        x_180 = x_179 * sigmoid_59
        x_179 = sigmoid_59 = None
        x_181 = torch.conv2d(
            x_159,
            l_self_modules_neck_modules_out_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_159 = l_self_modules_neck_modules_out_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_182 = torch.nn.functional.batch_norm(
            x_181,
            l_self_modules_neck_modules_out_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_out_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_out_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_out_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_181 = l_self_modules_neck_modules_out_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_out_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_out_convs_modules_1_modules_bn_parameters_weight_ = (
            l_self_modules_neck_modules_out_convs_modules_1_modules_bn_parameters_bias_
        ) = None
        sigmoid_60 = torch.sigmoid(x_182)
        x_183 = x_182 * sigmoid_60
        x_182 = sigmoid_60 = None
        x_184 = torch.conv2d(
            x_177,
            l_self_modules_neck_modules_out_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_neck_modules_out_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_neck_modules_out_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_out_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_out_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_out_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_184 = l_self_modules_neck_modules_out_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_out_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_out_convs_modules_2_modules_bn_parameters_weight_ = (
            l_self_modules_neck_modules_out_convs_modules_2_modules_bn_parameters_bias_
        ) = None
        sigmoid_61 = torch.sigmoid(x_185)
        x_186 = x_185 * sigmoid_61
        x_185 = sigmoid_61 = None
        x_187 = torch.conv2d(
            x_180,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_187 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_62 = torch.sigmoid(x_188)
        x_189 = x_188 * sigmoid_62
        x_188 = sigmoid_62 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_189 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_190 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_63 = torch.sigmoid(x_191)
        x_192 = x_191 * sigmoid_63
        x_191 = sigmoid_63 = None
        x_193 = torch.conv2d(
            x_180,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_193 = l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_64 = torch.sigmoid(x_194)
        x_195 = x_194 * sigmoid_64
        x_194 = sigmoid_64 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_196 = l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_65 = torch.sigmoid(x_197)
        x_198 = x_197 * sigmoid_65
        x_197 = sigmoid_65 = None
        x_199 = torch.conv2d(
            x_180,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_180 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        x_200 = torch.nn.functional.batch_norm(
            x_199,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_199 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_66 = torch.sigmoid(x_200)
        x_201 = x_200 * sigmoid_66
        x_200 = sigmoid_66 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_201 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_202 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_67 = torch.sigmoid(x_203)
        x_204 = x_203 * sigmoid_67
        x_203 = sigmoid_67 = None
        x_205 = torch.conv2d(
            x_204,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_204 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_conv_parameters_weight_ = (None)
        x_206 = torch.nn.functional.batch_norm(
            x_205,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_205 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_bias_ = (None)
        sigmoid_68 = torch.sigmoid(x_206)
        x_207 = x_206 * sigmoid_68
        x_206 = sigmoid_68 = None
        x_208 = torch.conv2d(
            x_207,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_207 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_conv_parameters_weight_ = (None)
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_208 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_bias_ = (None)
        sigmoid_69 = torch.sigmoid(x_209)
        x_210 = x_209 * sigmoid_69
        x_209 = sigmoid_69 = None
        conv2d_70 = torch.conv2d(
            x_192,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_192 = l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_bias_ = (None)
        conv2d_71 = torch.conv2d(
            x_198,
            l_self_modules_head_modules_head_module_modules_out_obj_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_obj_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_obj_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_obj_modules_0_parameters_bias_ = (None)
        conv2d_72 = torch.conv2d(
            x_198,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_198 = l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_ = (None)
        conv2d_73 = torch.conv2d(
            x_210,
            l_self_modules_head_modules_head_module_modules_out_kpt_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_kpt_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_kpt_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_kpt_modules_0_parameters_bias_ = (None)
        conv2d_74 = torch.conv2d(
            x_210,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_210 = l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_ = (None)
        x_211 = torch.conv2d(
            x_183,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_212 = torch.nn.functional.batch_norm(
            x_211,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_211 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_70 = torch.sigmoid(x_212)
        x_213 = x_212 * sigmoid_70
        x_212 = sigmoid_70 = None
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_213 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_214 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_71 = torch.sigmoid(x_215)
        x_216 = x_215 * sigmoid_71
        x_215 = sigmoid_71 = None
        x_217 = torch.conv2d(
            x_183,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_218 = torch.nn.functional.batch_norm(
            x_217,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_217 = l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_72 = torch.sigmoid(x_218)
        x_219 = x_218 * sigmoid_72
        x_218 = sigmoid_72 = None
        x_220 = torch.conv2d(
            x_219,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_219 = l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_220 = l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_73 = torch.sigmoid(x_221)
        x_222 = x_221 * sigmoid_73
        x_221 = sigmoid_73 = None
        x_223 = torch.conv2d(
            x_183,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        x_224 = torch.nn.functional.batch_norm(
            x_223,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_223 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_74 = torch.sigmoid(x_224)
        x_225 = x_224 * sigmoid_74
        x_224 = sigmoid_74 = None
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_227 = torch.nn.functional.batch_norm(
            x_226,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_226 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_75 = torch.sigmoid(x_227)
        x_228 = x_227 * sigmoid_75
        x_227 = sigmoid_75 = None
        x_229 = torch.conv2d(
            x_228,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_228 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_conv_parameters_weight_ = (None)
        x_230 = torch.nn.functional.batch_norm(
            x_229,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_229 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_bias_ = (None)
        sigmoid_76 = torch.sigmoid(x_230)
        x_231 = x_230 * sigmoid_76
        x_230 = sigmoid_76 = None
        x_232 = torch.conv2d(
            x_231,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_231 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_conv_parameters_weight_ = (None)
        x_233 = torch.nn.functional.batch_norm(
            x_232,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_232 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_bias_ = (None)
        sigmoid_77 = torch.sigmoid(x_233)
        x_234 = x_233 * sigmoid_77
        x_233 = sigmoid_77 = None
        conv2d_83 = torch.conv2d(
            x_216,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_216 = l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_bias_ = (None)
        conv2d_84 = torch.conv2d(
            x_222,
            l_self_modules_head_modules_head_module_modules_out_obj_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_obj_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_obj_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_obj_modules_1_parameters_bias_ = (None)
        conv2d_85 = torch.conv2d(
            x_222,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_222 = l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_ = (None)
        conv2d_86 = torch.conv2d(
            x_234,
            l_self_modules_head_modules_head_module_modules_out_kpt_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_kpt_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_kpt_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_kpt_modules_1_parameters_bias_ = (None)
        conv2d_87 = torch.conv2d(
            x_234,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_234 = l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_ = (None)
        x_235 = torch.conv2d(
            x_186,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_236 = torch.nn.functional.batch_norm(
            x_235,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_235 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_78 = torch.sigmoid(x_236)
        x_237 = x_236 * sigmoid_78
        x_236 = sigmoid_78 = None
        x_238 = torch.conv2d(
            x_237,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_237 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_239 = torch.nn.functional.batch_norm(
            x_238,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_238 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_79 = torch.sigmoid(x_239)
        x_240 = x_239 * sigmoid_79
        x_239 = sigmoid_79 = None
        x_241 = torch.conv2d(
            x_186,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_242 = torch.nn.functional.batch_norm(
            x_241,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_241 = l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_80 = torch.sigmoid(x_242)
        x_243 = x_242 * sigmoid_80
        x_242 = sigmoid_80 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_243 = l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_245 = torch.nn.functional.batch_norm(
            x_244,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_244 = l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_reg_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_81 = torch.sigmoid(x_245)
        x_246 = x_245 * sigmoid_81
        x_245 = sigmoid_81 = None
        x_247 = torch.conv2d(
            x_186,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_186 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_conv_parameters_weight_ = (None)
        x_248 = torch.nn.functional.batch_norm(
            x_247,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_247 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_82 = torch.sigmoid(x_248)
        x_249 = x_248 * sigmoid_82
        x_248 = sigmoid_82 = None
        x_250 = torch.conv2d(
            x_249,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_249 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_251 = torch.nn.functional.batch_norm(
            x_250,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_250 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_83 = torch.sigmoid(x_251)
        x_252 = x_251 * sigmoid_83
        x_251 = sigmoid_83 = None
        x_253 = torch.conv2d(
            x_252,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_252 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_conv_parameters_weight_ = (None)
        x_254 = torch.nn.functional.batch_norm(
            x_253,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_253 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_2_modules_bn_parameters_bias_ = (None)
        sigmoid_84 = torch.sigmoid(x_254)
        x_255 = x_254 * sigmoid_84
        x_254 = sigmoid_84 = None
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_255 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_conv_parameters_weight_ = (None)
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_256 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_2_modules_3_modules_bn_parameters_bias_ = (None)
        sigmoid_85 = torch.sigmoid(x_257)
        x_258 = x_257 * sigmoid_85
        x_257 = sigmoid_85 = None
        conv2d_96 = torch.conv2d(
            x_240,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_2_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_240 = l_self_modules_head_modules_head_module_modules_out_cls_modules_2_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_cls_modules_2_parameters_bias_ = (None)
        conv2d_97 = torch.conv2d(
            x_246,
            l_self_modules_head_modules_head_module_modules_out_obj_modules_2_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_obj_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_obj_modules_2_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_obj_modules_2_parameters_bias_ = (None)
        conv2d_98 = torch.conv2d(
            x_246,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_2_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_246 = l_self_modules_head_modules_head_module_modules_out_bbox_modules_2_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_bbox_modules_2_parameters_bias_ = (None)
        conv2d_99 = torch.conv2d(
            x_258,
            l_self_modules_head_modules_head_module_modules_out_kpt_modules_2_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_kpt_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_kpt_modules_2_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_kpt_modules_2_parameters_bias_ = (None)
        conv2d_100 = torch.conv2d(
            x_258,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_2_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_258 = l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_2_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_2_parameters_bias_ = (None)
        return (
            conv2d_70,
            conv2d_83,
            conv2d_96,
            conv2d_71,
            conv2d_84,
            conv2d_97,
            conv2d_72,
            conv2d_85,
            conv2d_98,
            conv2d_73,
            conv2d_86,
            conv2d_99,
            conv2d_74,
            conv2d_87,
            conv2d_100,
        )
