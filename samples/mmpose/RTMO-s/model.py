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
            x_51,
            l_self_modules_neck_modules_input_proj_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_neck_modules_input_proj_modules_0_modules_conv_parameters_weight_ = (None)
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = l_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_input_proj_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_neck_modules_input_proj_modules_0_modules_bn_parameters_bias_
        ) = None
        x_108 = torch.conv2d(
            x_81,
            l_self_modules_neck_modules_input_proj_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_neck_modules_input_proj_modules_1_modules_conv_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_input_proj_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_weight_ = (
            l_self_modules_neck_modules_input_proj_modules_1_modules_bn_parameters_bias_
        ) = None
        x_110 = torch.conv2d(
            x_105,
            l_self_modules_neck_modules_input_proj_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_105 = l_self_modules_neck_modules_input_proj_modules_2_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_input_proj_modules_2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_weight_ = (
            l_self_modules_neck_modules_input_proj_modules_2_modules_bn_parameters_bias_
        ) = None
        flatten = x_111.flatten(2)
        x_111 = None
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
        input_8 = torch._C._nn.linear(
            query_2,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_9 = torch._C._nn.gelu(input_8, approximate="none")
        input_8 = None
        input_10 = torch.nn.functional.dropout(input_9, 0.0, False, False)
        input_9 = None
        input_11 = torch._C._nn.linear(
            input_10,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_10 = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_neck_modules_encoder_modules_0_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_12 = torch.nn.functional.dropout(input_11, 0.0, False, False)
        input_11 = None
        output_1 = query_2 + input_12
        query_2 = input_12 = None
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
        x_112 = torch.conv2d(
            feat_high,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        feat_high = l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_114 = torch.nn.functional.silu(x_113, inplace=True)
        x_113 = None
        upsample_feat = torch.nn.functional.interpolate(
            x_114, scale_factor=2.0, mode="nearest"
        )
        cat_8 = torch.cat([upsample_feat, x_109], axis=1)
        upsample_feat = x_109 = None
        x_115 = torch.conv2d(
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
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_117 = torch.nn.functional.silu(x_116, inplace=True)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
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
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_120 = torch.conv2d(
            x_117,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_11 = x_119 + x_121
        x_119 = x_121 = None
        out_1 = add_11 + 0
        add_11 = None
        out_2 = torch.nn.functional.silu(out_1, inplace=True)
        out_1 = None
        x_122 = torch.conv2d(
            cat_8,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_8 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_124 = torch.nn.functional.silu(x_123, inplace=True)
        x_123 = None
        add_13 = out_2 + x_124
        out_2 = x_124 = None
        x_125 = torch.conv2d(
            add_13,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        add_13 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_127 = torch.nn.functional.silu(x_126, inplace=True)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_130 = torch.nn.functional.silu(x_129, inplace=True)
        x_129 = None
        upsample_feat_1 = torch.nn.functional.interpolate(
            x_130, scale_factor=2.0, mode="nearest"
        )
        cat_9 = torch.cat([upsample_feat_1, x_107], axis=1)
        upsample_feat_1 = x_107 = None
        x_131 = torch.conv2d(
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
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_133 = torch.nn.functional.silu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
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
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_136 = torch.conv2d(
            x_133,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_14 = x_135 + x_137
        x_135 = x_137 = None
        out_3 = add_14 + 0
        add_14 = None
        out_4 = torch.nn.functional.silu(out_3, inplace=True)
        out_3 = None
        x_138 = torch.conv2d(
            cat_9,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_9 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_140 = torch.nn.functional.silu(x_139, inplace=True)
        x_139 = None
        add_16 = out_4 + x_140
        out_4 = x_140 = None
        x_141 = torch.conv2d(
            add_16,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        add_16 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_fpn_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_143 = torch.nn.functional.silu(x_142, inplace=True)
        x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_neck_modules_downsample_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_neck_modules_downsample_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_neck_modules_downsample_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_146 = torch.nn.functional.silu(x_145, inplace=True)
        x_145 = None
        cat_10 = torch.cat([x_146, x_130], axis=1)
        x_146 = x_130 = None
        x_147 = torch.conv2d(
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
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_149 = torch.nn.functional.silu(x_148, inplace=True)
        x_148 = None
        x_150 = torch.conv2d(
            x_149,
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
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_150 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_152 = torch.conv2d(
            x_149,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_149 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_17 = x_151 + x_153
        x_151 = x_153 = None
        out_5 = add_17 + 0
        add_17 = None
        out_6 = torch.nn.functional.silu(out_5, inplace=True)
        out_5 = None
        x_154 = torch.conv2d(
            cat_10,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_10 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_156 = torch.nn.functional.silu(x_155, inplace=True)
        x_155 = None
        add_19 = out_6 + x_156
        out_6 = x_156 = None
        x_157 = torch.conv2d(
            add_19,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        add_19 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_159 = torch.nn.functional.silu(x_158, inplace=True)
        x_158 = None
        x_160 = torch.conv2d(
            x_159,
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
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_downsample_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_162 = torch.nn.functional.silu(x_161, inplace=True)
        x_161 = None
        cat_11 = torch.cat([x_162, x_114], axis=1)
        x_162 = x_114 = None
        x_163 = torch.conv2d(
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
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_163 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_165 = torch.nn.functional.silu(x_164, inplace=True)
        x_164 = None
        x_166 = torch.conv2d(
            x_165,
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
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_3x3_modules_bn_parameters_bias_ = (None)
        x_168 = torch.conv2d(
            x_165,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_165 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_conv_parameters_weight_ = (None)
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_bottlenecks_modules_0_modules_branch_1x1_modules_bn_parameters_bias_ = (None)
        add_20 = x_167 + x_169
        x_167 = x_169 = None
        out_7 = add_20 + 0
        add_20 = None
        out_8 = torch.nn.functional.silu(out_7, inplace=True)
        out_7 = None
        x_170 = torch.conv2d(
            cat_11,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_11 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_172 = torch.nn.functional.silu(x_171, inplace=True)
        x_171 = None
        add_22 = out_8 + x_172
        out_8 = x_172 = None
        x_173 = torch.conv2d(
            add_22,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        add_22 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_neck_modules_pan_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_175 = torch.nn.functional.silu(x_174, inplace=True)
        x_174 = None
        x_176 = torch.conv2d(
            x_159,
            l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_159 = l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_neck_modules_projector_modules_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_178 = torch.conv2d(
            x_175,
            l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_neck_modules_projector_modules_convs_modules_1_modules_bn_parameters_bias_ = (None)
        split = x_177.split(128, 1)
        x_177 = None
        cls_feat = split[0]
        reg_feat = split[1]
        split = None
        x_180 = torch.conv2d(
            cls_feat,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        cls_feat = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_180 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_35 = torch.sigmoid(x_181)
        x_182 = x_181 * sigmoid_35
        x_181 = sigmoid_35 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_183 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_36 = torch.sigmoid(x_184)
        x_185 = x_184 * sigmoid_36
        x_184 = sigmoid_36 = None
        x_186 = torch.conv2d(
            reg_feat,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        reg_feat = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_186 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_37 = torch.sigmoid(x_187)
        x_188 = x_187 * sigmoid_37
        x_187 = sigmoid_37 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_188 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_189 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_38 = torch.sigmoid(x_190)
        x_191 = x_190 * sigmoid_38
        x_190 = sigmoid_38 = None
        x_192 = torch.conv2d(
            x_191,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_191 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_conv_parameters_weight_ = (None)
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_192 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_2_modules_bn_parameters_bias_ = (None)
        sigmoid_39 = torch.sigmoid(x_193)
        x_194 = x_193 * sigmoid_39
        x_193 = sigmoid_39 = None
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_194 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_conv_parameters_weight_ = (None)
        x_196 = torch.nn.functional.batch_norm(
            x_195,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_195 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_0_modules_3_modules_bn_parameters_bias_ = (None)
        sigmoid_40 = torch.sigmoid(x_196)
        x_197 = x_196 * sigmoid_40
        x_196 = sigmoid_40 = None
        conv2d_70 = torch.conv2d(
            x_185,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_cls_modules_0_parameters_bias_ = (None)
        conv2d_71 = torch.conv2d(
            x_197,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_bbox_modules_0_parameters_bias_ = (None)
        conv2d_72 = torch.conv2d(
            x_197,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_0_parameters_bias_ = (None)
        conv2d_73 = torch.conv2d(
            x_197,
            l_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_pose_modules_0_parameters_bias_ = (None)
        split_1 = x_179.split(128, 1)
        x_179 = None
        cls_feat_1 = split_1[0]
        reg_feat_1 = split_1[1]
        split_1 = None
        x_198 = torch.conv2d(
            cls_feat_1,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        cls_feat_1 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_198 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_41 = torch.sigmoid(x_199)
        x_200 = x_199 * sigmoid_41
        x_199 = sigmoid_41 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_200 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_201 = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_cls_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_42 = torch.sigmoid(x_202)
        x_203 = x_202 * sigmoid_42
        x_202 = sigmoid_42 = None
        x_204 = torch.conv2d(
            reg_feat_1,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        reg_feat_1 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_204 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        sigmoid_43 = torch.sigmoid(x_205)
        x_206 = x_205 * sigmoid_43
        x_205 = sigmoid_43 = None
        x_207 = torch.conv2d(
            x_206,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_206 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_207 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        sigmoid_44 = torch.sigmoid(x_208)
        x_209 = x_208 * sigmoid_44
        x_208 = sigmoid_44 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_209 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_conv_parameters_weight_ = (None)
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_210 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_2_modules_bn_parameters_bias_ = (None)
        sigmoid_45 = torch.sigmoid(x_211)
        x_212 = x_211 * sigmoid_45
        x_211 = sigmoid_45 = None
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_212 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_conv_parameters_weight_ = (None)
        x_214 = torch.nn.functional.batch_norm(
            x_213,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_bias_,
            False,
            0.03,
            0.001,
        )
        x_213 = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_buffers_running_var_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_weight_ = l_self_modules_head_modules_head_module_modules_conv_pose_modules_1_modules_3_modules_bn_parameters_bias_ = (None)
        sigmoid_46 = torch.sigmoid(x_214)
        x_215 = x_214 * sigmoid_46
        x_214 = sigmoid_46 = None
        conv2d_80 = torch.conv2d(
            x_203,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_203 = l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_cls_modules_1_parameters_bias_ = (None)
        conv2d_81 = torch.conv2d(
            x_215,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_bbox_modules_1_parameters_bias_ = (None)
        conv2d_82 = torch.conv2d(
            x_215,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_kpt_vis_modules_1_parameters_bias_ = (None)
        conv2d_83 = torch.conv2d(
            x_215,
            l_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_weight_,
            l_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_215 = l_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_weight_ = l_self_modules_head_modules_head_module_modules_out_pose_modules_1_parameters_bias_ = (None)
        return (
            conv2d_70,
            conv2d_80,
            conv2d_71,
            conv2d_81,
            conv2d_72,
            conv2d_82,
            conv2d_73,
            conv2d_83,
        )
