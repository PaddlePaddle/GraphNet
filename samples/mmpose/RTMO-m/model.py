import torch

from torch import device


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
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_input_proj_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_input_proj_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_input_proj_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_sincos_pos_enc_buffers_dim_t_: torch.Tensor,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsample_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsample_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_projector_modules_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_projector_modules_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_bias_
        l_self_modules_neck_modules_input_proj_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_input_proj_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_weight_ = L_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_weight_
        l_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_neck_modules_input_proj_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_input_proj_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_neck_modules_input_proj_modules_2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_input_proj_modules_2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_neck_modules_sincos_pos_enc_buffers_dim_t_ = (
            L_self_modules_neck_modules_sincos_pos_enc_buffers_dim_t_
        )
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_weight_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_weight_
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_bias_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_bias_
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_0_parameters_weight_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_0_parameters_weight_
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_0_parameters_bias_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_0_parameters_bias_
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_1_parameters_weight_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_1_parameters_weight_
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_1_parameters_bias_ = L_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_1_parameters_bias_
        l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_bias_ = L_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_downsample_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_downsample_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_weight_ = L_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_weight_
        l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_bias_ = L_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_downsample_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_downsample_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_projector_modules_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_weight_ = L_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_weight_
        l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_bias_ = L_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_bias_
        l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_projector_modules_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_var_ = L_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_var_
        l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_weight_ = L_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_weight_
        l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_bias_ = L_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_bias_
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
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_bias_
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
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_
        l_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_weight_ = L_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_weight_
        l_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_bias_ = L_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_bias_
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
        x_19 = torch.conv2d(
            input_1,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_19 = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_6 = torch.sigmoid(x_20)
        x_21 = x_20 * sigmoid_6
        x_20 = sigmoid_6 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_22 = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage1_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_7 = torch.sigmoid(x_23)
        x_24 = x_23 * sigmoid_7
        x_23 = sigmoid_7 = None
        input_2 = x_24 + input_1
        x_24 = input_1 = None
        x_final = torch.cat((input_2, x_9), dim=1)
        input_2 = x_9 = None
        x_25 = torch.conv2d(
            x_final,
            l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final = l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_25 = l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage1_modules_1_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_8 = torch.sigmoid(x_26)
        x_27 = x_26 * sigmoid_8
        x_26 = sigmoid_8 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_backbone_modules_stage2_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_backbone_modules_stage2_modules_0_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_28 = l_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_stage2_modules_0_modules_bn_parameters_bias_
        ) = None
        sigmoid_9 = torch.sigmoid(x_29)
        x_30 = x_29 * sigmoid_9
        x_29 = sigmoid_9 = None
        x_31 = torch.conv2d(
            x_30,
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
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_31 = l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_short_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_10 = torch.sigmoid(x_32)
        x_33 = x_32 * sigmoid_10
        x_32 = sigmoid_10 = None
        x_34 = torch.conv2d(
            x_30,
            l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_34 = l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_main_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_11 = torch.sigmoid(x_35)
        x_36 = x_35 * sigmoid_11
        x_35 = sigmoid_11 = None
        x_37 = torch.conv2d(
            x_36,
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
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_37 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_12 = torch.sigmoid(x_38)
        x_39 = x_38 * sigmoid_12
        x_38 = sigmoid_12 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_40 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_13 = torch.sigmoid(x_41)
        x_42 = x_41 * sigmoid_13
        x_41 = sigmoid_13 = None
        input_3 = x_42 + x_36
        x_42 = x_36 = None
        x_43 = torch.conv2d(
            input_3,
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
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_43 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_14 = torch.sigmoid(x_44)
        x_45 = x_44 * sigmoid_14
        x_44 = sigmoid_14 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_46 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_15 = torch.sigmoid(x_47)
        x_48 = x_47 * sigmoid_15
        x_47 = sigmoid_15 = None
        input_4 = x_48 + input_3
        x_48 = input_3 = None
        x_49 = torch.conv2d(
            input_4,
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
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_49 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_16 = torch.sigmoid(x_50)
        x_51 = x_50 * sigmoid_16
        x_50 = sigmoid_16 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_52 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_17 = torch.sigmoid(x_53)
        x_54 = x_53 * sigmoid_17
        x_53 = sigmoid_17 = None
        input_5 = x_54 + input_4
        x_54 = input_4 = None
        x_55 = torch.conv2d(
            input_5,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_55 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_18 = torch.sigmoid(x_56)
        x_57 = x_56 * sigmoid_18
        x_56 = sigmoid_18 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_58 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_19 = torch.sigmoid(x_59)
        x_60 = x_59 * sigmoid_19
        x_59 = sigmoid_19 = None
        input_6 = x_60 + input_5
        x_60 = input_5 = None
        x_61 = torch.conv2d(
            input_6,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_61 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_20 = torch.sigmoid(x_62)
        x_63 = x_62 * sigmoid_20
        x_62 = sigmoid_20 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_64 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_21 = torch.sigmoid(x_65)
        x_66 = x_65 * sigmoid_21
        x_65 = sigmoid_21 = None
        input_7 = x_66 + input_6
        x_66 = input_6 = None
        x_67 = torch.conv2d(
            input_7,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_67 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_22 = torch.sigmoid(x_68)
        x_69 = x_68 * sigmoid_22
        x_68 = sigmoid_22 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_70 = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_23 = torch.sigmoid(x_71)
        x_72 = x_71 * sigmoid_23
        x_71 = sigmoid_23 = None
        input_8 = x_72 + input_7
        x_72 = input_7 = None
        x_final_1 = torch.cat((input_8, x_33), dim=1)
        input_8 = x_33 = None
        x_73 = torch.conv2d(
            x_final_1,
            l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final_1 = l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_73 = l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage2_modules_1_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_24 = torch.sigmoid(x_74)
        x_75 = x_74 * sigmoid_24
        x_74 = sigmoid_24 = None
        x_76 = torch.conv2d(
            x_75,
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
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_76 = l_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_stage3_modules_0_modules_bn_parameters_bias_
        ) = None
        sigmoid_25 = torch.sigmoid(x_77)
        x_78 = x_77 * sigmoid_25
        x_77 = sigmoid_25 = None
        x_79 = torch.conv2d(
            x_78,
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
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_79 = l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_short_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_26 = torch.sigmoid(x_80)
        x_81 = x_80 * sigmoid_26
        x_80 = sigmoid_26 = None
        x_82 = torch.conv2d(
            x_78,
            l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_conv_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_82 = l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_main_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_27 = torch.sigmoid(x_83)
        x_84 = x_83 * sigmoid_27
        x_83 = sigmoid_27 = None
        x_85 = torch.conv2d(
            x_84,
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
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_85 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_28 = torch.sigmoid(x_86)
        x_87 = x_86 * sigmoid_28
        x_86 = sigmoid_28 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_87 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_88 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_29 = torch.sigmoid(x_89)
        x_90 = x_89 * sigmoid_29
        x_89 = sigmoid_29 = None
        input_9 = x_90 + x_84
        x_90 = x_84 = None
        x_91 = torch.conv2d(
            input_9,
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
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_91 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_30 = torch.sigmoid(x_92)
        x_93 = x_92 * sigmoid_30
        x_92 = sigmoid_30 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_94 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_31 = torch.sigmoid(x_95)
        x_96 = x_95 * sigmoid_31
        x_95 = sigmoid_31 = None
        input_10 = x_96 + input_9
        x_96 = input_9 = None
        x_97 = torch.conv2d(
            input_10,
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
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_97 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_32 = torch.sigmoid(x_98)
        x_99 = x_98 * sigmoid_32
        x_98 = sigmoid_32 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_100 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_33 = torch.sigmoid(x_101)
        x_102 = x_101 * sigmoid_33
        x_101 = sigmoid_33 = None
        input_11 = x_102 + input_10
        x_102 = input_10 = None
        x_103 = torch.conv2d(
            input_11,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_103 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_34 = torch.sigmoid(x_104)
        x_105 = x_104 * sigmoid_34
        x_104 = sigmoid_34 = None
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_105 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_106 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_35 = torch.sigmoid(x_107)
        x_108 = x_107 * sigmoid_35
        x_107 = sigmoid_35 = None
        input_12 = x_108 + input_11
        x_108 = input_11 = None
        x_109 = torch.conv2d(
            input_12,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_109 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_36 = torch.sigmoid(x_110)
        x_111 = x_110 * sigmoid_36
        x_110 = sigmoid_36 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_112 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_37 = torch.sigmoid(x_113)
        x_114 = x_113 * sigmoid_37
        x_113 = sigmoid_37 = None
        input_13 = x_114 + input_12
        x_114 = input_12 = None
        x_115 = torch.conv2d(
            input_13,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_115 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_38 = torch.sigmoid(x_116)
        x_117 = x_116 * sigmoid_38
        x_116 = sigmoid_38 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_118 = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_39 = torch.sigmoid(x_119)
        x_120 = x_119 * sigmoid_39
        x_119 = sigmoid_39 = None
        input_14 = x_120 + input_13
        x_120 = input_13 = None
        x_final_2 = torch.cat((input_14, x_81), dim=1)
        input_14 = x_81 = None
        x_121 = torch.conv2d(
            x_final_2,
            l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final_2 = l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_121 = l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage3_modules_1_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_40 = torch.sigmoid(x_122)
        x_123 = x_122 * sigmoid_40
        x_122 = sigmoid_40 = None
        x_124 = torch.conv2d(
            x_123,
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
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_124 = l_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_stage4_modules_0_modules_bn_parameters_bias_
        ) = None
        sigmoid_41 = torch.sigmoid(x_125)
        x_126 = x_125 * sigmoid_41
        x_125 = sigmoid_41 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_126 = l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_127 = l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_42 = torch.sigmoid(x_128)
        x_129 = x_128 * sigmoid_42
        x_128 = sigmoid_42 = None
        max_pool2d = torch.nn.functional.max_pool2d(
            x_129, 5, 1, 2, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_1 = torch.nn.functional.max_pool2d(
            x_129, 9, 1, 4, 1, ceil_mode=False, return_indices=False
        )
        max_pool2d_2 = torch.nn.functional.max_pool2d(
            x_129, 13, 1, 6, 1, ceil_mode=False, return_indices=False
        )
        cat_4 = torch.cat([x_129, max_pool2d, max_pool2d_1, max_pool2d_2], dim=1)
        x_129 = max_pool2d = max_pool2d_1 = max_pool2d_2 = None
        x_130 = torch.conv2d(
            cat_4,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_4 = l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_130 = l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_43 = torch.sigmoid(x_131)
        x_132 = x_131 * sigmoid_43
        x_131 = sigmoid_43 = None
        x_133 = torch.conv2d(
            x_132,
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
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_133 = l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_short_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_44 = torch.sigmoid(x_134)
        x_135 = x_134 * sigmoid_44
        x_134 = sigmoid_44 = None
        x_136 = torch.conv2d(
            x_132,
            l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_132 = l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_136 = l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_main_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_45 = torch.sigmoid(x_137)
        x_138 = x_137 * sigmoid_45
        x_137 = sigmoid_45 = None
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_138 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_139 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_46 = torch.sigmoid(x_140)
        x_141 = x_140 * sigmoid_46
        x_140 = sigmoid_46 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_142 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_47 = torch.sigmoid(x_143)
        x_144 = x_143 * sigmoid_47
        x_143 = sigmoid_47 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_144 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (None)
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_145 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        sigmoid_48 = torch.sigmoid(x_146)
        x_147 = x_146 * sigmoid_48
        x_146 = sigmoid_48 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_148 = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        sigmoid_49 = torch.sigmoid(x_149)
        x_150 = x_149 * sigmoid_49
        x_149 = sigmoid_49 = None
        x_final_3 = torch.cat((x_150, x_135), dim=1)
        x_150 = x_135 = None
        x_151 = torch.conv2d(
            x_final_3,
            l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_final_3 = l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_151 = l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stage4_modules_2_modules_final_conv_modules_bn_parameters_bias_ = (None)
        sigmoid_50 = torch.sigmoid(x_152)
        x_153 = x_152 * sigmoid_50
        x_152 = sigmoid_50 = None
        x_154 = torch.conv2d(
            x_75,
            l_self_modules_neck_modules_input_proj_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_neck_modules_input_proj_modules_0_modules_conv_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_bias_
        ) = None
        x_156 = torch.conv2d(
            x_123,
            l_self_modules_neck_modules_input_proj_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_123 = l_self_modules_neck_modules_input_proj_modules_1_modules_conv_parameters_weight_ = (None)
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = l_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_weight_ = (
            l_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_bias_
        ) = None
        x_158 = torch.conv2d(
            x_153,
            l_self_modules_neck_modules_input_proj_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_neck_modules_input_proj_modules_2_modules_conv_parameters_weight_ = (None)
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = l_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_weight_ = (
            l_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_bias_
        ) = None
        flatten = x_159.flatten(2)
        x_159 = None
        permute = flatten.permute(0, 2, 1)
        flatten = None
        src_flatten = permute.contiguous()
        permute = None
        arange = torch.arange(
            8, dtype=torch.float32, device=device(type="cuda", index=0)
        )
        arange_1 = torch.arange(
            8, dtype=torch.float32, device=device(type="cuda", index=0)
        )
        meshgrid = torch.functional.meshgrid(arange, arange_1)
        arange = arange_1 = None
        grid_h = meshgrid[0]
        grid_w = meshgrid[1]
        meshgrid = None
        grid_h_1 = grid_h.flatten()
        grid_h = None
        grid_w_1 = grid_w.flatten()
        grid_w = None
        dim_t = l_self_modules_neck_modules_sincos_pos_enc_buffers_dim_t_.reshape(1, -1)
        l_self_modules_neck_modules_sincos_pos_enc_buffers_dim_t_ = None
        unsqueeze = grid_h_1.unsqueeze(-1)
        grid_h_1 = None
        freq_h = unsqueeze / dim_t
        unsqueeze = None
        unsqueeze_1 = grid_w_1.unsqueeze(-1)
        grid_w_1 = None
        freq_w = unsqueeze_1 / dim_t
        unsqueeze_1 = dim_t = None
        cos = freq_h.cos()
        sin = freq_h.sin()
        freq_h = None
        pos_enc_h = torch.cat((cos, sin), dim=-1)
        cos = sin = None
        cos_1 = freq_w.cos()
        sin_1 = freq_w.sin()
        freq_w = None
        pos_enc_w = torch.cat((cos_1, sin_1), dim=-1)
        cos_1 = sin_1 = None
        pos_enc = torch.stack((pos_enc_h, pos_enc_w), dim=-1)
        pos_enc_h = pos_enc_w = None
        transpose = pos_enc.transpose(-1, -2)
        pos_enc = None
        pos_enc_1 = transpose.reshape(1, 64, -1)
        transpose = None
        query = src_flatten + pos_enc_1
        key = src_flatten + pos_enc_1
        pos_enc_1 = None
        query_1 = query.transpose(0, 1)
        query = None
        key_1 = key.transpose(0, 1)
        key = None
        value = src_flatten.transpose(0, 1)
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(
            query_1,
            key_1,
            value,
            256,
            8,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_weight_,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query_1 = (
            key_1
        ) = (
            value
        ) = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_weight_ = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_parameters_in_proj_bias_ = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_self_attn_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output = multi_head_attention_forward[0]
        multi_head_attention_forward = None
        out = attn_output.transpose(0, 1)
        attn_output = None
        dropout = torch.nn.functional.dropout(out, 0.0, False, False)
        out = None
        dropout_1 = torch.nn.functional.dropout(dropout, 0.0, False, False)
        dropout = None
        output = src_flatten + dropout_1
        src_flatten = dropout_1 = None
        query_2 = torch.nn.functional.layer_norm(
            output,
            (256,),
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_0_parameters_weight_,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_0_parameters_bias_,
            1e-05,
        )
        output = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_0_parameters_weight_ = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_0_parameters_bias_ = (None)
        input_15 = torch._C._nn.linear(
            query_2,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_16 = torch._C._nn.gelu(input_15, approximate="none")
        input_15 = None
        input_17 = torch.nn.functional.dropout(input_16, 0.0, False, False)
        input_16 = None
        input_18 = torch._C._nn.linear(
            input_17,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_17 = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_19 = torch.nn.functional.dropout(input_18, 0.0, False, False)
        input_18 = None
        output_1 = query_2 + input_19
        query_2 = input_19 = None
        query_3 = torch.nn.functional.layer_norm(
            output_1,
            (256,),
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_1_parameters_weight_,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_1_parameters_bias_,
            1e-05,
        )
        output_1 = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_1_parameters_weight_ = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_norms_modules_1_parameters_bias_ = (None)
        permute_1 = query_3.permute(0, 2, 1)
        query_3 = None
        contiguous_1 = permute_1.contiguous()
        permute_1 = None
        feat_high = contiguous_1.view([-1, 256, 8, 8])
        contiguous_1 = None
        x_160 = torch.conv2d(
            feat_high,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        feat_high = l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_162 = torch.nn.functional.silu(x_161, inplace=True)
        x_161 = None
        upsample_feat = torch.nn.functional.interpolate(
            x_162, scale_factor=2.0, mode="nearest"
        )
        cat_8 = torch.cat([upsample_feat, x_157], axis=1)
        upsample_feat = x_157 = None
        x_163 = torch.conv2d(
            cat_8,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_163 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_165 = torch.nn.functional.silu(x_164, inplace=True)
        x_164 = None
        x_166 = torch.conv2d(
            x_165,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_ = (
            None
        )
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_168 = torch.conv2d(
            x_165,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_165 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_18 = x_167 + x_169
        x_167 = x_169 = None
        out_1 = add_18 + 0
        add_18 = None
        out_2 = torch.nn.functional.silu(out_1, inplace=True)
        out_1 = None
        x_170 = torch.conv2d(
            out_2,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_ = (
            None
        )
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_172 = torch.conv2d(
            out_2,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_2 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_20 = x_171 + x_173
        x_171 = x_173 = None
        out_3 = add_20 + 0
        add_20 = None
        out_4 = torch.nn.functional.silu(out_3, inplace=True)
        out_3 = None
        x_174 = torch.conv2d(
            cat_8,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_8 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_176 = torch.nn.functional.silu(x_175, inplace=True)
        x_175 = None
        add_22 = out_4 + x_176
        out_4 = x_176 = None
        x_177 = torch.conv2d(
            add_22,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        add_22 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_177 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_179 = torch.nn.functional.silu(x_178, inplace=True)
        x_178 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_182 = torch.nn.functional.silu(x_181, inplace=True)
        x_181 = None
        upsample_feat_1 = torch.nn.functional.interpolate(
            x_182, scale_factor=2.0, mode="nearest"
        )
        cat_9 = torch.cat([upsample_feat_1, x_155], axis=1)
        upsample_feat_1 = x_155 = None
        x_183 = torch.conv2d(
            cat_9,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_185 = torch.nn.functional.silu(x_184, inplace=True)
        x_184 = None
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_ = (
            None
        )
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_188 = torch.conv2d(
            x_185,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_189 = torch.nn.functional.batch_norm(
            x_188,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_188 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_23 = x_187 + x_189
        x_187 = x_189 = None
        out_5 = add_23 + 0
        add_23 = None
        out_6 = torch.nn.functional.silu(out_5, inplace=True)
        out_5 = None
        x_190 = torch.conv2d(
            out_6,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_ = (
            None
        )
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_192 = torch.conv2d(
            out_6,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_6 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_192 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_25 = x_191 + x_193
        x_191 = x_193 = None
        out_7 = add_25 + 0
        add_25 = None
        out_8 = torch.nn.functional.silu(out_7, inplace=True)
        out_7 = None
        x_194 = torch.conv2d(
            cat_9,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_9 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_196 = torch.nn.functional.silu(x_195, inplace=True)
        x_195 = None
        add_27 = out_8 + x_196
        out_8 = x_196 = None
        x_197 = torch.conv2d(
            add_27,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        add_27 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_197 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_199 = torch.nn.functional.silu(x_198, inplace=True)
        x_198 = None
        x_200 = torch.conv2d(
            x_199,
            l_self_modules_neck_modules_downsample_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_199 = l_self_modules_neck_modules_downsample_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_201 = torch.nn.functional.batch_norm(
            x_200,
            l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_200 = l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_202 = torch.nn.functional.silu(x_201, inplace=True)
        x_201 = None
        cat_10 = torch.cat([x_202, x_182], axis=1)
        x_202 = x_182 = None
        x_203 = torch.conv2d(
            cat_10,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_204 = torch.nn.functional.batch_norm(
            x_203,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_203 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_205 = torch.nn.functional.silu(x_204, inplace=True)
        x_204 = None
        x_206 = torch.conv2d(
            x_205,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_ = (
            None
        )
        x_207 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_206 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_208 = torch.conv2d(
            x_205,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_208 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_28 = x_207 + x_209
        x_207 = x_209 = None
        out_9 = add_28 + 0
        add_28 = None
        out_10 = torch.nn.functional.silu(out_9, inplace=True)
        out_9 = None
        x_210 = torch.conv2d(
            out_10,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_ = (
            None
        )
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_210 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_212 = torch.conv2d(
            out_10,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_10 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_30 = x_211 + x_213
        x_211 = x_213 = None
        out_11 = add_30 + 0
        add_30 = None
        out_12 = torch.nn.functional.silu(out_11, inplace=True)
        out_11 = None
        x_214 = torch.conv2d(
            cat_10,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_10 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_214 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_216 = torch.nn.functional.silu(x_215, inplace=True)
        x_215 = None
        add_32 = out_12 + x_216
        out_12 = x_216 = None
        x_217 = torch.conv2d(
            add_32,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        add_32 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_218 = torch.nn.functional.batch_norm(
            x_217,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_217 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_219 = torch.nn.functional.silu(x_218, inplace=True)
        x_218 = None
        x_220 = torch.conv2d(
            x_219,
            l_self_modules_neck_modules_downsample_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_downsample_convs_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_220 = l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_222 = torch.nn.functional.silu(x_221, inplace=True)
        x_221 = None
        cat_11 = torch.cat([x_222, x_162], axis=1)
        x_222 = x_162 = None
        x_223 = torch.conv2d(
            cat_11,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_224 = torch.nn.functional.batch_norm(
            x_223,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_223 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_225 = torch.nn.functional.silu(x_224, inplace=True)
        x_224 = None
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_conv_parameters_weight_ = (
            None
        )
        x_227 = torch.nn.functional.batch_norm(
            x_226,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_226 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_228 = torch.conv2d(
            x_225,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_229 = torch.nn.functional.batch_norm(
            x_228,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_228 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_33 = x_227 + x_229
        x_227 = x_229 = None
        out_13 = add_33 + 0
        add_33 = None
        out_14 = torch.nn.functional.silu(out_13, inplace=True)
        out_13 = None
        x_230 = torch.conv2d(
            out_14,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_conv_parameters_weight_ = (
            None
        )
        x_231 = torch.nn.functional.batch_norm(
            x_230,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_230 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_232 = torch.conv2d(
            out_14,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_14 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_233 = torch.nn.functional.batch_norm(
            x_232,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_232 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_1_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_35 = x_231 + x_233
        x_231 = x_233 = None
        out_15 = add_35 + 0
        add_35 = None
        out_16 = torch.nn.functional.silu(out_15, inplace=True)
        out_15 = None
        x_234 = torch.conv2d(
            cat_11,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_11 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_236 = torch.nn.functional.silu(x_235, inplace=True)
        x_235 = None
        add_37 = out_16 + x_236
        out_16 = x_236 = None
        x_237 = torch.conv2d(
            add_37,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        add_37 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_238 = torch.nn.functional.batch_norm(
            x_237,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_237 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_239 = torch.nn.functional.silu(x_238, inplace=True)
        x_238 = None
        x_240 = torch.conv2d(
            x_219,
            l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_219 = l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_241 = torch.nn.functional.batch_norm(
            x_240,
            l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_240 = l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_242 = torch.conv2d(
            x_239,
            l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_239 = l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_243 = torch.nn.functional.batch_norm(
            x_242,
            l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_242 = l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_bias_ = (None)
        split = x_241.split(192, 1)
        x_241 = None
        cls_feat = split[0]
        reg_feat = split[1]
        split = None
        x_244 = torch.conv2d(
            cls_feat,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        cls_feat = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        x_245 = torch.nn.functional.batch_norm(
            x_244,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_244 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_51 = torch.sigmoid(x_245)
        x_246 = x_245 * sigmoid_51
        x_245 = sigmoid_51 = None
        x_247 = torch.conv2d(
            x_246,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_246 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_248 = torch.nn.functional.batch_norm(
            x_247,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_247 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_52 = torch.sigmoid(x_248)
        x_249 = x_248 * sigmoid_52
        x_248 = sigmoid_52 = None
        x_250 = torch.conv2d(
            reg_feat,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        reg_feat = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        x_251 = torch.nn.functional.batch_norm(
            x_250,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_250 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_53 = torch.sigmoid(x_251)
        x_252 = x_251 * sigmoid_53
        x_251 = sigmoid_53 = None
        x_253 = torch.conv2d(
            x_252,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_252 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_254 = torch.nn.functional.batch_norm(
            x_253,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_253 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_54 = torch.sigmoid(x_254)
        x_255 = x_254 * sigmoid_54
        x_254 = sigmoid_54 = None
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_255 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_conv_parameters_weight_ = (None)
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_256 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_bias_ = (None)
        sigmoid_55 = torch.sigmoid(x_257)
        x_258 = x_257 * sigmoid_55
        x_257 = sigmoid_55 = None
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_258 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_conv_parameters_weight_ = (None)
        x_260 = torch.nn.functional.batch_norm(
            x_259,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_259 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_bias_ = (None)
        sigmoid_56 = torch.sigmoid(x_260)
        x_261 = x_260 * sigmoid_56
        x_260 = sigmoid_56 = None
        conv2d_94 = torch.conv2d(
            x_249,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_249 = l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_bias_ = (None)
        conv2d_95 = torch.conv2d(
            x_261,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_ = (None)
        conv2d_96 = torch.conv2d(
            x_261,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_ = (None)
        conv2d_97 = torch.conv2d(
            x_261,
            l_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_261 = l_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_bias_ = (None)
        split_1 = x_243.split(192, 1)
        x_243 = None
        cls_feat_1 = split_1[0]
        reg_feat_1 = split_1[1]
        split_1 = None
        x_262 = torch.conv2d(
            cls_feat_1,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        cls_feat_1 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        x_263 = torch.nn.functional.batch_norm(
            x_262,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_262 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_57 = torch.sigmoid(x_263)
        x_264 = x_263 * sigmoid_57
        x_263 = sigmoid_57 = None
        x_265 = torch.conv2d(
            x_264,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_264 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_266 = torch.nn.functional.batch_norm(
            x_265,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_265 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_58 = torch.sigmoid(x_266)
        x_267 = x_266 * sigmoid_58
        x_266 = sigmoid_58 = None
        x_268 = torch.conv2d(
            reg_feat_1,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        reg_feat_1 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        x_269 = torch.nn.functional.batch_norm(
            x_268,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_268 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_59 = torch.sigmoid(x_269)
        x_270 = x_269 * sigmoid_59
        x_269 = sigmoid_59 = None
        x_271 = torch.conv2d(
            x_270,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_270 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_272 = torch.nn.functional.batch_norm(
            x_271,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_271 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_60 = torch.sigmoid(x_272)
        x_273 = x_272 * sigmoid_60
        x_272 = sigmoid_60 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_273 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_conv_parameters_weight_ = (None)
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_274 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_bias_ = (None)
        sigmoid_61 = torch.sigmoid(x_275)
        x_276 = x_275 * sigmoid_61
        x_275 = sigmoid_61 = None
        x_277 = torch.conv2d(
            x_276,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_276 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_conv_parameters_weight_ = (None)
        x_278 = torch.nn.functional.batch_norm(
            x_277,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_277 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_bias_ = (None)
        sigmoid_62 = torch.sigmoid(x_278)
        x_279 = x_278 * sigmoid_62
        x_278 = sigmoid_62 = None
        conv2d_104 = torch.conv2d(
            x_267,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_267 = l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_bias_ = (None)
        conv2d_105 = torch.conv2d(
            x_279,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_ = (None)
        conv2d_106 = torch.conv2d(
            x_279,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_ = (None)
        conv2d_107 = torch.conv2d(
            x_279,
            l_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_279 = l_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_bias_ = (None)
        return (
            conv2d_94,
            conv2d_104,
            conv2d_95,
            conv2d_105,
            conv2d_96,
            conv2d_106,
            conv2d_97,
            conv2d_107,
        )
