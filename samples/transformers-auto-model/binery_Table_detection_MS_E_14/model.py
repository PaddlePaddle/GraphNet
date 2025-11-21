import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_pixel_values_: torch.Tensor,
        L_pixel_mask_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_weight_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_bias_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_input_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_input_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_query_position_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_pixel_values_ = L_pixel_values_
        l_pixel_mask_ = L_pixel_mask_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_weight_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_weight_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_bias_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_bias_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_input_projection_parameters_weight_ = (
            L_self_modules_input_projection_parameters_weight_
        )
        l_self_modules_input_projection_parameters_bias_ = (
            L_self_modules_input_projection_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = (
            L_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        )
        l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        l_self_modules_query_position_embeddings_parameters_weight_ = (
            L_self_modules_query_position_embeddings_parameters_weight_
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_ = (
            L_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = L_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_
        l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = L_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_
        l_self_modules_decoder_modules_layernorm_parameters_weight_ = (
            L_self_modules_decoder_modules_layernorm_parameters_weight_
        )
        l_self_modules_decoder_modules_layernorm_parameters_bias_ = (
            L_self_modules_decoder_modules_layernorm_parameters_bias_
        )
        x = torch.conv2d(
            l_pixel_values_,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            1,
        )
        l_pixel_values_ = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_conv1_parameters_weight_ = (None)
        weight = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_weight_ = (
            None
        )
        bias = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_bias_ = (
            None
        )
        running_var = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_running_var_ = (
            None
        )
        running_mean = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_bn1_buffers_running_mean_ = (
            None
        )
        add = running_var + 1e-05
        running_var = None
        rsqrt = add.rsqrt()
        add = None
        scale = weight * rsqrt
        weight = rsqrt = None
        mul_1 = running_mean * scale
        running_mean = None
        bias_1 = bias - mul_1
        bias = mul_1 = None
        mul_2 = x * scale
        x = scale = None
        x_1 = mul_2 + bias_1
        mul_2 = bias_1 = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.nn.functional.max_pool2d(
            x_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_2 = None
        x_4 = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        weight_1 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_weight_ = (
            None
        )
        bias_2 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_bias_ = (
            None
        )
        running_var_1 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = (
            None
        )
        running_mean_1 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = (
            None
        )
        add_2 = running_var_1 + 1e-05
        running_var_1 = None
        rsqrt_1 = add_2.rsqrt()
        add_2 = None
        scale_1 = weight_1 * rsqrt_1
        weight_1 = rsqrt_1 = None
        mul_4 = running_mean_1 * scale_1
        running_mean_1 = None
        bias_3 = bias_2 - mul_4
        bias_2 = mul_4 = None
        mul_5 = x_4 * scale_1
        x_4 = scale_1 = None
        x_5 = mul_5 + bias_3
        mul_5 = bias_3 = None
        x_6 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_conv2_parameters_weight_ = (None)
        weight_2 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_weight_ = (
            None
        )
        bias_4 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_bias_ = (
            None
        )
        running_var_2 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = (
            None
        )
        running_mean_2 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = (
            None
        )
        add_4 = running_var_2 + 1e-05
        running_var_2 = None
        rsqrt_2 = add_4.rsqrt()
        add_4 = None
        scale_2 = weight_2 * rsqrt_2
        weight_2 = rsqrt_2 = None
        mul_7 = running_mean_2 * scale_2
        running_mean_2 = None
        bias_5 = bias_4 - mul_7
        bias_4 = mul_7 = None
        mul_8 = x_7 * scale_2
        x_7 = scale_2 = None
        x_8 = mul_8 + bias_5
        mul_8 = bias_5 = None
        x_8 += x_3
        x_9 = x_8
        x_8 = x_3 = None
        x_10 = torch.nn.functional.relu(x_9, inplace=True)
        x_9 = None
        x_11 = torch.conv2d(
            x_10,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        weight_3 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_weight_ = (
            None
        )
        bias_6 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_bias_ = (
            None
        )
        running_var_3 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = (
            None
        )
        running_mean_3 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = (
            None
        )
        add_6 = running_var_3 + 1e-05
        running_var_3 = None
        rsqrt_3 = add_6.rsqrt()
        add_6 = None
        scale_3 = weight_3 * rsqrt_3
        weight_3 = rsqrt_3 = None
        mul_10 = running_mean_3 * scale_3
        running_mean_3 = None
        bias_7 = bias_6 - mul_10
        bias_6 = mul_10 = None
        mul_11 = x_11 * scale_3
        x_11 = scale_3 = None
        x_12 = mul_11 + bias_7
        mul_11 = bias_7 = None
        x_13 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_conv2_parameters_weight_ = (None)
        weight_4 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_weight_ = (
            None
        )
        bias_8 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_bias_ = (
            None
        )
        running_var_4 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = (
            None
        )
        running_mean_4 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = (
            None
        )
        add_8 = running_var_4 + 1e-05
        running_var_4 = None
        rsqrt_4 = add_8.rsqrt()
        add_8 = None
        scale_4 = weight_4 * rsqrt_4
        weight_4 = rsqrt_4 = None
        mul_13 = running_mean_4 * scale_4
        running_mean_4 = None
        bias_9 = bias_8 - mul_13
        bias_8 = mul_13 = None
        mul_14 = x_14 * scale_4
        x_14 = scale_4 = None
        x_15 = mul_14 + bias_9
        mul_14 = bias_9 = None
        x_15 += x_10
        x_16 = x_15
        x_15 = x_10 = None
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        weight_5 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_weight_ = (
            None
        )
        bias_10 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_bias_ = (
            None
        )
        running_var_5 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = (
            None
        )
        running_mean_5 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = (
            None
        )
        add_10 = running_var_5 + 1e-05
        running_var_5 = None
        rsqrt_5 = add_10.rsqrt()
        add_10 = None
        scale_5 = weight_5 * rsqrt_5
        weight_5 = rsqrt_5 = None
        mul_16 = running_mean_5 * scale_5
        running_mean_5 = None
        bias_11 = bias_10 - mul_16
        bias_10 = mul_16 = None
        mul_17 = x_18 * scale_5
        x_18 = scale_5 = None
        x_19 = mul_17 + bias_11
        mul_17 = bias_11 = None
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_conv2_parameters_weight_ = (None)
        weight_6 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_weight_ = (
            None
        )
        bias_12 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_bias_ = (
            None
        )
        running_var_6 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = (
            None
        )
        running_mean_6 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = (
            None
        )
        add_12 = running_var_6 + 1e-05
        running_var_6 = None
        rsqrt_6 = add_12.rsqrt()
        add_12 = None
        scale_6 = weight_6 * rsqrt_6
        weight_6 = rsqrt_6 = None
        mul_19 = running_mean_6 * scale_6
        running_mean_6 = None
        bias_13 = bias_12 - mul_19
        bias_12 = mul_19 = None
        mul_20 = x_21 * scale_6
        x_21 = scale_6 = None
        x_22 = mul_20 + bias_13
        mul_20 = bias_13 = None
        input_1 = torch.conv2d(
            x_17,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        weight_7 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_weight_ = (
            None
        )
        bias_14 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_bias_ = (
            None
        )
        running_var_7 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = (
            None
        )
        running_mean_7 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = (
            None
        )
        add_14 = running_var_7 + 1e-05
        running_var_7 = None
        rsqrt_7 = add_14.rsqrt()
        add_14 = None
        scale_7 = weight_7 * rsqrt_7
        weight_7 = rsqrt_7 = None
        mul_22 = running_mean_7 * scale_7
        running_mean_7 = None
        bias_15 = bias_14 - mul_22
        bias_14 = mul_22 = None
        mul_23 = input_1 * scale_7
        input_1 = scale_7 = None
        input_2 = mul_23 + bias_15
        mul_23 = bias_15 = None
        x_22 += input_2
        x_23 = x_22
        x_22 = input_2 = None
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        weight_8 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_weight_ = (
            None
        )
        bias_16 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_bias_ = (
            None
        )
        running_var_8 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = (
            None
        )
        running_mean_8 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = (
            None
        )
        add_16 = running_var_8 + 1e-05
        running_var_8 = None
        rsqrt_8 = add_16.rsqrt()
        add_16 = None
        scale_8 = weight_8 * rsqrt_8
        weight_8 = rsqrt_8 = None
        mul_25 = running_mean_8 * scale_8
        running_mean_8 = None
        bias_17 = bias_16 - mul_25
        bias_16 = mul_25 = None
        mul_26 = x_25 * scale_8
        x_25 = scale_8 = None
        x_26 = mul_26 + bias_17
        mul_26 = bias_17 = None
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_conv2_parameters_weight_ = (None)
        weight_9 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_weight_ = (
            None
        )
        bias_18 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_bias_ = (
            None
        )
        running_var_9 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = (
            None
        )
        running_mean_9 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = (
            None
        )
        add_18 = running_var_9 + 1e-05
        running_var_9 = None
        rsqrt_9 = add_18.rsqrt()
        add_18 = None
        scale_9 = weight_9 * rsqrt_9
        weight_9 = rsqrt_9 = None
        mul_28 = running_mean_9 * scale_9
        running_mean_9 = None
        bias_19 = bias_18 - mul_28
        bias_18 = mul_28 = None
        mul_29 = x_28 * scale_9
        x_28 = scale_9 = None
        x_29 = mul_29 + bias_19
        mul_29 = bias_19 = None
        x_29 += x_24
        x_30 = x_29
        x_29 = x_24 = None
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        weight_10 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_weight_ = (
            None
        )
        bias_20 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_bias_ = (
            None
        )
        running_var_10 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = (
            None
        )
        running_mean_10 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = (
            None
        )
        add_20 = running_var_10 + 1e-05
        running_var_10 = None
        rsqrt_10 = add_20.rsqrt()
        add_20 = None
        scale_10 = weight_10 * rsqrt_10
        weight_10 = rsqrt_10 = None
        mul_31 = running_mean_10 * scale_10
        running_mean_10 = None
        bias_21 = bias_20 - mul_31
        bias_20 = mul_31 = None
        mul_32 = x_32 * scale_10
        x_32 = scale_10 = None
        x_33 = mul_32 + bias_21
        mul_32 = bias_21 = None
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_conv2_parameters_weight_ = (None)
        weight_11 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_weight_ = (
            None
        )
        bias_22 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_bias_ = (
            None
        )
        running_var_11 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = (
            None
        )
        running_mean_11 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = (
            None
        )
        add_22 = running_var_11 + 1e-05
        running_var_11 = None
        rsqrt_11 = add_22.rsqrt()
        add_22 = None
        scale_11 = weight_11 * rsqrt_11
        weight_11 = rsqrt_11 = None
        mul_34 = running_mean_11 * scale_11
        running_mean_11 = None
        bias_23 = bias_22 - mul_34
        bias_22 = mul_34 = None
        mul_35 = x_35 * scale_11
        x_35 = scale_11 = None
        x_36 = mul_35 + bias_23
        mul_35 = bias_23 = None
        input_3 = torch.conv2d(
            x_31,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        weight_12 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_weight_ = (
            None
        )
        bias_24 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_bias_ = (
            None
        )
        running_var_12 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = (
            None
        )
        running_mean_12 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = (
            None
        )
        add_24 = running_var_12 + 1e-05
        running_var_12 = None
        rsqrt_12 = add_24.rsqrt()
        add_24 = None
        scale_12 = weight_12 * rsqrt_12
        weight_12 = rsqrt_12 = None
        mul_37 = running_mean_12 * scale_12
        running_mean_12 = None
        bias_25 = bias_24 - mul_37
        bias_24 = mul_37 = None
        mul_38 = input_3 * scale_12
        input_3 = scale_12 = None
        input_4 = mul_38 + bias_25
        mul_38 = bias_25 = None
        x_36 += input_4
        x_37 = x_36
        x_36 = input_4 = None
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        weight_13 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_weight_ = (
            None
        )
        bias_26 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_bias_ = (
            None
        )
        running_var_13 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = (
            None
        )
        running_mean_13 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = (
            None
        )
        add_26 = running_var_13 + 1e-05
        running_var_13 = None
        rsqrt_13 = add_26.rsqrt()
        add_26 = None
        scale_13 = weight_13 * rsqrt_13
        weight_13 = rsqrt_13 = None
        mul_40 = running_mean_13 * scale_13
        running_mean_13 = None
        bias_27 = bias_26 - mul_40
        bias_26 = mul_40 = None
        mul_41 = x_39 * scale_13
        x_39 = scale_13 = None
        x_40 = mul_41 + bias_27
        mul_41 = bias_27 = None
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_conv2_parameters_weight_ = (None)
        weight_14 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_weight_ = (
            None
        )
        bias_28 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_bias_ = (
            None
        )
        running_var_14 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = (
            None
        )
        running_mean_14 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = (
            None
        )
        add_28 = running_var_14 + 1e-05
        running_var_14 = None
        rsqrt_14 = add_28.rsqrt()
        add_28 = None
        scale_14 = weight_14 * rsqrt_14
        weight_14 = rsqrt_14 = None
        mul_43 = running_mean_14 * scale_14
        running_mean_14 = None
        bias_29 = bias_28 - mul_43
        bias_28 = mul_43 = None
        mul_44 = x_42 * scale_14
        x_42 = scale_14 = None
        x_43 = mul_44 + bias_29
        mul_44 = bias_29 = None
        x_43 += x_38
        x_44 = x_43
        x_43 = x_38 = None
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        weight_15 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_weight_ = (
            None
        )
        bias_30 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_bias_ = (
            None
        )
        running_var_15 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = (
            None
        )
        running_mean_15 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = (
            None
        )
        add_30 = running_var_15 + 1e-05
        running_var_15 = None
        rsqrt_15 = add_30.rsqrt()
        add_30 = None
        scale_15 = weight_15 * rsqrt_15
        weight_15 = rsqrt_15 = None
        mul_46 = running_mean_15 * scale_15
        running_mean_15 = None
        bias_31 = bias_30 - mul_46
        bias_30 = mul_46 = None
        mul_47 = x_46 * scale_15
        x_46 = scale_15 = None
        x_47 = mul_47 + bias_31
        mul_47 = bias_31 = None
        x_48 = torch.nn.functional.relu(x_47, inplace=True)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_conv2_parameters_weight_ = (None)
        weight_16 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_weight_ = (
            None
        )
        bias_32 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_bias_ = (
            None
        )
        running_var_16 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = (
            None
        )
        running_mean_16 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = (
            None
        )
        add_32 = running_var_16 + 1e-05
        running_var_16 = None
        rsqrt_16 = add_32.rsqrt()
        add_32 = None
        scale_16 = weight_16 * rsqrt_16
        weight_16 = rsqrt_16 = None
        mul_49 = running_mean_16 * scale_16
        running_mean_16 = None
        bias_33 = bias_32 - mul_49
        bias_32 = mul_49 = None
        mul_50 = x_49 * scale_16
        x_49 = scale_16 = None
        x_50 = mul_50 + bias_33
        mul_50 = bias_33 = None
        input_5 = torch.conv2d(
            x_45,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        weight_17 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_weight_ = (
            None
        )
        bias_34 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_bias_ = (
            None
        )
        running_var_17 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = (
            None
        )
        running_mean_17 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = (
            None
        )
        add_34 = running_var_17 + 1e-05
        running_var_17 = None
        rsqrt_17 = add_34.rsqrt()
        add_34 = None
        scale_17 = weight_17 * rsqrt_17
        weight_17 = rsqrt_17 = None
        mul_52 = running_mean_17 * scale_17
        running_mean_17 = None
        bias_35 = bias_34 - mul_52
        bias_34 = mul_52 = None
        mul_53 = input_5 * scale_17
        input_5 = scale_17 = None
        input_6 = mul_53 + bias_35
        mul_53 = bias_35 = None
        x_50 += input_6
        x_51 = x_50
        x_50 = input_6 = None
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        weight_18 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_weight_ = (
            None
        )
        bias_36 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_bias_ = (
            None
        )
        running_var_18 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = (
            None
        )
        running_mean_18 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = (
            None
        )
        add_36 = running_var_18 + 1e-05
        running_var_18 = None
        rsqrt_18 = add_36.rsqrt()
        add_36 = None
        scale_18 = weight_18 * rsqrt_18
        weight_18 = rsqrt_18 = None
        mul_55 = running_mean_18 * scale_18
        running_mean_18 = None
        bias_37 = bias_36 - mul_55
        bias_36 = mul_55 = None
        mul_56 = x_53 * scale_18
        x_53 = scale_18 = None
        x_54 = mul_56 + bias_37
        mul_56 = bias_37 = None
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_conv2_parameters_weight_ = (None)
        weight_19 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_weight_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_weight_ = (
            None
        )
        bias_38 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_bias_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_bias_ = (
            None
        )
        running_var_19 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_running_var_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = (
            None
        )
        running_mean_19 = l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_running_mean_.reshape(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_conv_encoder_modules_model_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = (
            None
        )
        add_38 = running_var_19 + 1e-05
        running_var_19 = None
        rsqrt_19 = add_38.rsqrt()
        add_38 = None
        scale_19 = weight_19 * rsqrt_19
        weight_19 = rsqrt_19 = None
        mul_58 = running_mean_19 * scale_19
        running_mean_19 = None
        bias_39 = bias_38 - mul_58
        bias_38 = mul_58 = None
        mul_59 = x_56 * scale_19
        x_56 = scale_19 = None
        x_57 = mul_59 + bias_39
        mul_59 = bias_39 = None
        x_57 += x_52
        x_58 = x_57
        x_57 = x_52 = None
        x_59 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        getitem = l_pixel_mask_[None]
        float_1 = getitem.float()
        getitem = None
        interpolate = torch.nn.functional.interpolate(float_1, size=(200, 200))
        float_1 = None
        to = interpolate.to(torch.bool)
        interpolate = None
        mask = to[0]
        to = None
        getitem_2 = l_pixel_mask_[None]
        float_2 = getitem_2.float()
        getitem_2 = None
        interpolate_1 = torch.nn.functional.interpolate(float_2, size=(100, 100))
        float_2 = None
        to_1 = interpolate_1.to(torch.bool)
        interpolate_1 = None
        mask_1 = to_1[0]
        to_1 = None
        getitem_4 = l_pixel_mask_[None]
        float_3 = getitem_4.float()
        getitem_4 = None
        interpolate_2 = torch.nn.functional.interpolate(float_3, size=(50, 50))
        float_3 = None
        to_2 = interpolate_2.to(torch.bool)
        interpolate_2 = None
        mask_2 = to_2[0]
        to_2 = None
        getitem_6 = l_pixel_mask_[None]
        l_pixel_mask_ = None
        float_4 = getitem_6.float()
        getitem_6 = None
        interpolate_3 = torch.nn.functional.interpolate(float_4, size=(25, 25))
        float_4 = None
        to_3 = interpolate_3.to(torch.bool)
        interpolate_3 = None
        mask_3 = to_3[0]
        to_3 = None
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        mask = None
        getitem_8 = y_embed[
            (slice(None, None, None), slice(-1, None, None), slice(None, None, None))
        ]
        add_40 = getitem_8 + 1e-06
        getitem_8 = None
        truediv = y_embed / add_40
        y_embed = add_40 = None
        y_embed_1 = truediv * 6.283185307179586
        truediv = None
        getitem_9 = x_embed[
            (slice(None, None, None), slice(None, None, None), slice(-1, None, None))
        ]
        add_41 = getitem_9 + 1e-06
        getitem_9 = None
        truediv_1 = x_embed / add_41
        x_embed = add_41 = None
        x_embed_1 = truediv_1 * 6.283185307179586
        truediv_1 = None
        arange = torch.arange(
            128, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        dim_t = arange.float()
        arange = None
        div = torch.div(dim_t, 2, rounding_mode="floor")
        dim_t = None
        mul_62 = 2 * div
        div = None
        truediv_2 = mul_62 / 128
        mul_62 = None
        dim_t_1 = 10000**truediv_2
        truediv_2 = None
        getitem_10 = x_embed_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        x_embed_1 = None
        pos_x = getitem_10 / dim_t_1
        getitem_10 = None
        getitem_11 = y_embed_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        y_embed_1 = None
        pos_y = getitem_11 / dim_t_1
        getitem_11 = dim_t_1 = None
        getitem_12 = pos_x[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(0, None, 2),
            )
        ]
        sin = getitem_12.sin()
        getitem_12 = None
        getitem_13 = pos_x[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        pos_x = None
        cos = getitem_13.cos()
        getitem_13 = None
        stack = torch.stack((sin, cos), dim=4)
        sin = cos = None
        pos_x_1 = stack.flatten(3)
        stack = None
        getitem_14 = pos_y[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(0, None, 2),
            )
        ]
        sin_1 = getitem_14.sin()
        getitem_14 = None
        getitem_15 = pos_y[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        pos_y = None
        cos_1 = getitem_15.cos()
        getitem_15 = None
        stack_1 = torch.stack((sin_1, cos_1), dim=4)
        sin_1 = cos_1 = None
        pos_y_1 = stack_1.flatten(3)
        stack_1 = None
        cat = torch.cat((pos_y_1, pos_x_1), dim=3)
        pos_y_1 = pos_x_1 = None
        pos = cat.permute(0, 3, 1, 2)
        cat = None
        to_4 = pos.to(torch.float32)
        pos = to_4 = None
        y_embed_2 = mask_1.cumsum(1, dtype=torch.float32)
        x_embed_2 = mask_1.cumsum(2, dtype=torch.float32)
        mask_1 = None
        getitem_16 = y_embed_2[
            (slice(None, None, None), slice(-1, None, None), slice(None, None, None))
        ]
        add_42 = getitem_16 + 1e-06
        getitem_16 = None
        truediv_5 = y_embed_2 / add_42
        y_embed_2 = add_42 = None
        y_embed_3 = truediv_5 * 6.283185307179586
        truediv_5 = None
        getitem_17 = x_embed_2[
            (slice(None, None, None), slice(None, None, None), slice(-1, None, None))
        ]
        add_43 = getitem_17 + 1e-06
        getitem_17 = None
        truediv_6 = x_embed_2 / add_43
        x_embed_2 = add_43 = None
        x_embed_3 = truediv_6 * 6.283185307179586
        truediv_6 = None
        arange_1 = torch.arange(
            128, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        dim_t_2 = arange_1.float()
        arange_1 = None
        div_1 = torch.div(dim_t_2, 2, rounding_mode="floor")
        dim_t_2 = None
        mul_65 = 2 * div_1
        div_1 = None
        truediv_7 = mul_65 / 128
        mul_65 = None
        dim_t_3 = 10000**truediv_7
        truediv_7 = None
        getitem_18 = x_embed_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        x_embed_3 = None
        pos_x_2 = getitem_18 / dim_t_3
        getitem_18 = None
        getitem_19 = y_embed_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        y_embed_3 = None
        pos_y_2 = getitem_19 / dim_t_3
        getitem_19 = dim_t_3 = None
        getitem_20 = pos_x_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(0, None, 2),
            )
        ]
        sin_2 = getitem_20.sin()
        getitem_20 = None
        getitem_21 = pos_x_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        pos_x_2 = None
        cos_2 = getitem_21.cos()
        getitem_21 = None
        stack_2 = torch.stack((sin_2, cos_2), dim=4)
        sin_2 = cos_2 = None
        pos_x_3 = stack_2.flatten(3)
        stack_2 = None
        getitem_22 = pos_y_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(0, None, 2),
            )
        ]
        sin_3 = getitem_22.sin()
        getitem_22 = None
        getitem_23 = pos_y_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        pos_y_2 = None
        cos_3 = getitem_23.cos()
        getitem_23 = None
        stack_3 = torch.stack((sin_3, cos_3), dim=4)
        sin_3 = cos_3 = None
        pos_y_3 = stack_3.flatten(3)
        stack_3 = None
        cat_1 = torch.cat((pos_y_3, pos_x_3), dim=3)
        pos_y_3 = pos_x_3 = None
        pos_1 = cat_1.permute(0, 3, 1, 2)
        cat_1 = None
        to_5 = pos_1.to(torch.float32)
        pos_1 = to_5 = None
        y_embed_4 = mask_2.cumsum(1, dtype=torch.float32)
        x_embed_4 = mask_2.cumsum(2, dtype=torch.float32)
        mask_2 = None
        getitem_24 = y_embed_4[
            (slice(None, None, None), slice(-1, None, None), slice(None, None, None))
        ]
        add_44 = getitem_24 + 1e-06
        getitem_24 = None
        truediv_10 = y_embed_4 / add_44
        y_embed_4 = add_44 = None
        y_embed_5 = truediv_10 * 6.283185307179586
        truediv_10 = None
        getitem_25 = x_embed_4[
            (slice(None, None, None), slice(None, None, None), slice(-1, None, None))
        ]
        add_45 = getitem_25 + 1e-06
        getitem_25 = None
        truediv_11 = x_embed_4 / add_45
        x_embed_4 = add_45 = None
        x_embed_5 = truediv_11 * 6.283185307179586
        truediv_11 = None
        arange_2 = torch.arange(
            128, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        dim_t_4 = arange_2.float()
        arange_2 = None
        div_2 = torch.div(dim_t_4, 2, rounding_mode="floor")
        dim_t_4 = None
        mul_68 = 2 * div_2
        div_2 = None
        truediv_12 = mul_68 / 128
        mul_68 = None
        dim_t_5 = 10000**truediv_12
        truediv_12 = None
        getitem_26 = x_embed_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        x_embed_5 = None
        pos_x_4 = getitem_26 / dim_t_5
        getitem_26 = None
        getitem_27 = y_embed_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        y_embed_5 = None
        pos_y_4 = getitem_27 / dim_t_5
        getitem_27 = dim_t_5 = None
        getitem_28 = pos_x_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(0, None, 2),
            )
        ]
        sin_4 = getitem_28.sin()
        getitem_28 = None
        getitem_29 = pos_x_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        pos_x_4 = None
        cos_4 = getitem_29.cos()
        getitem_29 = None
        stack_4 = torch.stack((sin_4, cos_4), dim=4)
        sin_4 = cos_4 = None
        pos_x_5 = stack_4.flatten(3)
        stack_4 = None
        getitem_30 = pos_y_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(0, None, 2),
            )
        ]
        sin_5 = getitem_30.sin()
        getitem_30 = None
        getitem_31 = pos_y_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        pos_y_4 = None
        cos_5 = getitem_31.cos()
        getitem_31 = None
        stack_5 = torch.stack((sin_5, cos_5), dim=4)
        sin_5 = cos_5 = None
        pos_y_5 = stack_5.flatten(3)
        stack_5 = None
        cat_2 = torch.cat((pos_y_5, pos_x_5), dim=3)
        pos_y_5 = pos_x_5 = None
        pos_2 = cat_2.permute(0, 3, 1, 2)
        cat_2 = None
        to_6 = pos_2.to(torch.float32)
        pos_2 = to_6 = None
        y_embed_6 = mask_3.cumsum(1, dtype=torch.float32)
        x_embed_6 = mask_3.cumsum(2, dtype=torch.float32)
        getitem_32 = y_embed_6[
            (slice(None, None, None), slice(-1, None, None), slice(None, None, None))
        ]
        add_46 = getitem_32 + 1e-06
        getitem_32 = None
        truediv_15 = y_embed_6 / add_46
        y_embed_6 = add_46 = None
        y_embed_7 = truediv_15 * 6.283185307179586
        truediv_15 = None
        getitem_33 = x_embed_6[
            (slice(None, None, None), slice(None, None, None), slice(-1, None, None))
        ]
        add_47 = getitem_33 + 1e-06
        getitem_33 = None
        truediv_16 = x_embed_6 / add_47
        x_embed_6 = add_47 = None
        x_embed_7 = truediv_16 * 6.283185307179586
        truediv_16 = None
        arange_3 = torch.arange(
            128, dtype=torch.int64, device=device(type="cuda", index=0)
        )
        dim_t_6 = arange_3.float()
        arange_3 = None
        div_3 = torch.div(dim_t_6, 2, rounding_mode="floor")
        dim_t_6 = None
        mul_71 = 2 * div_3
        div_3 = None
        truediv_17 = mul_71 / 128
        mul_71 = None
        dim_t_7 = 10000**truediv_17
        truediv_17 = None
        getitem_34 = x_embed_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        x_embed_7 = None
        pos_x_6 = getitem_34 / dim_t_7
        getitem_34 = None
        getitem_35 = y_embed_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        y_embed_7 = None
        pos_y_6 = getitem_35 / dim_t_7
        getitem_35 = dim_t_7 = None
        getitem_36 = pos_x_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(0, None, 2),
            )
        ]
        sin_6 = getitem_36.sin()
        getitem_36 = None
        getitem_37 = pos_x_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        pos_x_6 = None
        cos_6 = getitem_37.cos()
        getitem_37 = None
        stack_6 = torch.stack((sin_6, cos_6), dim=4)
        sin_6 = cos_6 = None
        pos_x_7 = stack_6.flatten(3)
        stack_6 = None
        getitem_38 = pos_y_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(0, None, 2),
            )
        ]
        sin_7 = getitem_38.sin()
        getitem_38 = None
        getitem_39 = pos_y_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, 2),
            )
        ]
        pos_y_6 = None
        cos_7 = getitem_39.cos()
        getitem_39 = None
        stack_7 = torch.stack((sin_7, cos_7), dim=4)
        sin_7 = cos_7 = None
        pos_y_7 = stack_7.flatten(3)
        stack_7 = None
        cat_3 = torch.cat((pos_y_7, pos_x_7), dim=3)
        pos_y_7 = pos_x_7 = None
        pos_3 = cat_3.permute(0, 3, 1, 2)
        cat_3 = None
        to_7 = pos_3.to(torch.float32)
        pos_3 = None
        projected_feature_map = torch.conv2d(
            x_59,
            l_self_modules_input_projection_parameters_weight_,
            l_self_modules_input_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = (
            l_self_modules_input_projection_parameters_weight_
        ) = l_self_modules_input_projection_parameters_bias_ = None
        flatten_8 = projected_feature_map.flatten(2)
        projected_feature_map = None
        flattened_features = flatten_8.permute(0, 2, 1)
        flatten_8 = None
        flatten_9 = to_7.flatten(2)
        to_7 = None
        object_queries = flatten_9.permute(0, 2, 1)
        flatten_9 = None
        flattened_mask = mask_3.flatten(1)
        mask_3 = None
        hidden_states = torch.nn.functional.dropout(
            flattened_features, p=0.1, training=False
        )
        flattened_features = None
        getitem_40 = flattened_mask[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        expand = getitem_40.expand(1, 1, 625, 625)
        getitem_40 = None
        expanded_mask = expand.to(torch.float32)
        expand = None
        tensor = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask = tensor - expanded_mask
        tensor = expanded_mask = None
        to_9 = inverted_mask.to(torch.bool)
        attention_mask = inverted_mask.masked_fill(to_9, -3.4028234663852886e38)
        inverted_mask = to_9 = None
        hidden_states_1 = hidden_states + object_queries
        linear = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states = linear * 0.1767766952966369
        linear = None
        linear_1 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_1 = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view = linear_1.view(1, -1, 8, 32)
        linear_1 = None
        transpose = view.transpose(1, 2)
        view = None
        key_states = transpose.contiguous()
        transpose = None
        linear_2 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_1 = linear_2.view(1, -1, 8, 32)
        linear_2 = None
        transpose_1 = view_1.transpose(1, 2)
        view_1 = None
        value_states = transpose_1.contiguous()
        transpose_1 = None
        view_2 = query_states.view(1, 625, 8, 32)
        query_states = None
        transpose_2 = view_2.transpose(1, 2)
        view_2 = None
        contiguous_2 = transpose_2.contiguous()
        transpose_2 = None
        query_states_1 = contiguous_2.view(8, -1, 32)
        contiguous_2 = None
        key_states_1 = key_states.view(8, -1, 32)
        key_states = None
        value_states_1 = value_states.view(8, -1, 32)
        value_states = None
        transpose_3 = key_states_1.transpose(1, 2)
        key_states_1 = None
        attn_weights = torch.bmm(query_states_1, transpose_3)
        query_states_1 = transpose_3 = None
        view_6 = attn_weights.view(1, 8, 625, 625)
        attn_weights = None
        attn_weights_1 = view_6 + attention_mask
        view_6 = None
        attn_weights_2 = attn_weights_1.view(8, 625, 625)
        attn_weights_1 = None
        attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim=-1)
        attn_weights_2 = None
        attn_probs = torch.nn.functional.dropout(attn_weights_3, p=0.0, training=False)
        attn_weights_3 = None
        attn_output = torch.bmm(attn_probs, value_states_1)
        attn_probs = value_states_1 = None
        attn_output_1 = attn_output.view(1, 8, 625, 32)
        attn_output = None
        attn_output_2 = attn_output_1.transpose(1, 2)
        attn_output_1 = None
        attn_output_3 = attn_output_2.reshape(1, 625, 256)
        attn_output_2 = None
        attn_output_4 = torch._C._nn.linear(
            attn_output_3,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_3 = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_2 = torch.nn.functional.dropout(
            attn_output_4, p=0.1, training=False
        )
        attn_output_4 = None
        hidden_states_3 = hidden_states + hidden_states_2
        hidden_states = hidden_states_2 = None
        hidden_states_4 = torch.nn.functional.layer_norm(
            hidden_states_3,
            (256,),
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_3 = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            hidden_states_4,
            l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        ) = None
        hidden_states_5 = torch.nn.functional.relu(linear_4, inplace=False)
        linear_4 = None
        hidden_states_6 = torch.nn.functional.dropout(
            hidden_states_5, p=0.0, training=False
        )
        hidden_states_5 = None
        hidden_states_7 = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_6 = l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        ) = None
        hidden_states_8 = torch.nn.functional.dropout(
            hidden_states_7, p=0.1, training=False
        )
        hidden_states_7 = None
        hidden_states_9 = hidden_states_4 + hidden_states_8
        hidden_states_4 = hidden_states_8 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (256,),
            l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_9 = l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_11 = hidden_states_10 + object_queries
        linear_6 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_2 = linear_6 * 0.1767766952966369
        linear_6 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_11 = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_9 = linear_7.view(1, -1, 8, 32)
        linear_7 = None
        transpose_5 = view_9.transpose(1, 2)
        view_9 = None
        key_states_2 = transpose_5.contiguous()
        transpose_5 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_10 = linear_8.view(1, -1, 8, 32)
        linear_8 = None
        transpose_6 = view_10.transpose(1, 2)
        view_10 = None
        value_states_2 = transpose_6.contiguous()
        transpose_6 = None
        view_11 = query_states_2.view(1, 625, 8, 32)
        query_states_2 = None
        transpose_7 = view_11.transpose(1, 2)
        view_11 = None
        contiguous_5 = transpose_7.contiguous()
        transpose_7 = None
        query_states_3 = contiguous_5.view(8, -1, 32)
        contiguous_5 = None
        key_states_3 = key_states_2.view(8, -1, 32)
        key_states_2 = None
        value_states_3 = value_states_2.view(8, -1, 32)
        value_states_2 = None
        transpose_8 = key_states_3.transpose(1, 2)
        key_states_3 = None
        attn_weights_4 = torch.bmm(query_states_3, transpose_8)
        query_states_3 = transpose_8 = None
        view_15 = attn_weights_4.view(1, 8, 625, 625)
        attn_weights_4 = None
        attn_weights_5 = view_15 + attention_mask
        view_15 = None
        attn_weights_6 = attn_weights_5.view(8, 625, 625)
        attn_weights_5 = None
        attn_weights_7 = torch.nn.functional.softmax(attn_weights_6, dim=-1)
        attn_weights_6 = None
        attn_probs_1 = torch.nn.functional.dropout(
            attn_weights_7, p=0.0, training=False
        )
        attn_weights_7 = None
        attn_output_5 = torch.bmm(attn_probs_1, value_states_3)
        attn_probs_1 = value_states_3 = None
        attn_output_6 = attn_output_5.view(1, 8, 625, 32)
        attn_output_5 = None
        attn_output_7 = attn_output_6.transpose(1, 2)
        attn_output_6 = None
        attn_output_8 = attn_output_7.reshape(1, 625, 256)
        attn_output_7 = None
        attn_output_9 = torch._C._nn.linear(
            attn_output_8,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_8 = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_12 = torch.nn.functional.dropout(
            attn_output_9, p=0.1, training=False
        )
        attn_output_9 = None
        hidden_states_13 = hidden_states_10 + hidden_states_12
        hidden_states_10 = hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.layer_norm(
            hidden_states_13,
            (256,),
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_13 = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_10 = torch._C._nn.linear(
            hidden_states_14,
            l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        ) = None
        hidden_states_15 = torch.nn.functional.relu(linear_10, inplace=False)
        linear_10 = None
        hidden_states_16 = torch.nn.functional.dropout(
            hidden_states_15, p=0.0, training=False
        )
        hidden_states_15 = None
        hidden_states_17 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_16 = l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        ) = None
        hidden_states_18 = torch.nn.functional.dropout(
            hidden_states_17, p=0.1, training=False
        )
        hidden_states_17 = None
        hidden_states_19 = hidden_states_14 + hidden_states_18
        hidden_states_14 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (256,),
            l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_19 = l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_21 = hidden_states_20 + object_queries
        linear_12 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_4 = linear_12 * 0.1767766952966369
        linear_12 = None
        linear_13 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_21 = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_18 = linear_13.view(1, -1, 8, 32)
        linear_13 = None
        transpose_10 = view_18.transpose(1, 2)
        view_18 = None
        key_states_4 = transpose_10.contiguous()
        transpose_10 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_19 = linear_14.view(1, -1, 8, 32)
        linear_14 = None
        transpose_11 = view_19.transpose(1, 2)
        view_19 = None
        value_states_4 = transpose_11.contiguous()
        transpose_11 = None
        view_20 = query_states_4.view(1, 625, 8, 32)
        query_states_4 = None
        transpose_12 = view_20.transpose(1, 2)
        view_20 = None
        contiguous_8 = transpose_12.contiguous()
        transpose_12 = None
        query_states_5 = contiguous_8.view(8, -1, 32)
        contiguous_8 = None
        key_states_5 = key_states_4.view(8, -1, 32)
        key_states_4 = None
        value_states_5 = value_states_4.view(8, -1, 32)
        value_states_4 = None
        transpose_13 = key_states_5.transpose(1, 2)
        key_states_5 = None
        attn_weights_8 = torch.bmm(query_states_5, transpose_13)
        query_states_5 = transpose_13 = None
        view_24 = attn_weights_8.view(1, 8, 625, 625)
        attn_weights_8 = None
        attn_weights_9 = view_24 + attention_mask
        view_24 = None
        attn_weights_10 = attn_weights_9.view(8, 625, 625)
        attn_weights_9 = None
        attn_weights_11 = torch.nn.functional.softmax(attn_weights_10, dim=-1)
        attn_weights_10 = None
        attn_probs_2 = torch.nn.functional.dropout(
            attn_weights_11, p=0.0, training=False
        )
        attn_weights_11 = None
        attn_output_10 = torch.bmm(attn_probs_2, value_states_5)
        attn_probs_2 = value_states_5 = None
        attn_output_11 = attn_output_10.view(1, 8, 625, 32)
        attn_output_10 = None
        attn_output_12 = attn_output_11.transpose(1, 2)
        attn_output_11 = None
        attn_output_13 = attn_output_12.reshape(1, 625, 256)
        attn_output_12 = None
        attn_output_14 = torch._C._nn.linear(
            attn_output_13,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_13 = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_22 = torch.nn.functional.dropout(
            attn_output_14, p=0.1, training=False
        )
        attn_output_14 = None
        hidden_states_23 = hidden_states_20 + hidden_states_22
        hidden_states_20 = hidden_states_22 = None
        hidden_states_24 = torch.nn.functional.layer_norm(
            hidden_states_23,
            (256,),
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_23 = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_16 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_
        ) = None
        hidden_states_25 = torch.nn.functional.relu(linear_16, inplace=False)
        linear_16 = None
        hidden_states_26 = torch.nn.functional.dropout(
            hidden_states_25, p=0.0, training=False
        )
        hidden_states_25 = None
        hidden_states_27 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_26 = l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_
        ) = None
        hidden_states_28 = torch.nn.functional.dropout(
            hidden_states_27, p=0.1, training=False
        )
        hidden_states_27 = None
        hidden_states_29 = hidden_states_24 + hidden_states_28
        hidden_states_24 = hidden_states_28 = None
        hidden_states_30 = torch.nn.functional.layer_norm(
            hidden_states_29,
            (256,),
            l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_29 = l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_31 = hidden_states_30 + object_queries
        linear_18 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_6 = linear_18 * 0.1767766952966369
        linear_18 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_31 = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_27 = linear_19.view(1, -1, 8, 32)
        linear_19 = None
        transpose_15 = view_27.transpose(1, 2)
        view_27 = None
        key_states_6 = transpose_15.contiguous()
        transpose_15 = None
        linear_20 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_28 = linear_20.view(1, -1, 8, 32)
        linear_20 = None
        transpose_16 = view_28.transpose(1, 2)
        view_28 = None
        value_states_6 = transpose_16.contiguous()
        transpose_16 = None
        view_29 = query_states_6.view(1, 625, 8, 32)
        query_states_6 = None
        transpose_17 = view_29.transpose(1, 2)
        view_29 = None
        contiguous_11 = transpose_17.contiguous()
        transpose_17 = None
        query_states_7 = contiguous_11.view(8, -1, 32)
        contiguous_11 = None
        key_states_7 = key_states_6.view(8, -1, 32)
        key_states_6 = None
        value_states_7 = value_states_6.view(8, -1, 32)
        value_states_6 = None
        transpose_18 = key_states_7.transpose(1, 2)
        key_states_7 = None
        attn_weights_12 = torch.bmm(query_states_7, transpose_18)
        query_states_7 = transpose_18 = None
        view_33 = attn_weights_12.view(1, 8, 625, 625)
        attn_weights_12 = None
        attn_weights_13 = view_33 + attention_mask
        view_33 = None
        attn_weights_14 = attn_weights_13.view(8, 625, 625)
        attn_weights_13 = None
        attn_weights_15 = torch.nn.functional.softmax(attn_weights_14, dim=-1)
        attn_weights_14 = None
        attn_probs_3 = torch.nn.functional.dropout(
            attn_weights_15, p=0.0, training=False
        )
        attn_weights_15 = None
        attn_output_15 = torch.bmm(attn_probs_3, value_states_7)
        attn_probs_3 = value_states_7 = None
        attn_output_16 = attn_output_15.view(1, 8, 625, 32)
        attn_output_15 = None
        attn_output_17 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_18 = attn_output_17.reshape(1, 625, 256)
        attn_output_17 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_18 = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_32 = torch.nn.functional.dropout(
            attn_output_19, p=0.1, training=False
        )
        attn_output_19 = None
        hidden_states_33 = hidden_states_30 + hidden_states_32
        hidden_states_30 = hidden_states_32 = None
        hidden_states_34 = torch.nn.functional.layer_norm(
            hidden_states_33,
            (256,),
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_33 = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_22 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_
        ) = None
        hidden_states_35 = torch.nn.functional.relu(linear_22, inplace=False)
        linear_22 = None
        hidden_states_36 = torch.nn.functional.dropout(
            hidden_states_35, p=0.0, training=False
        )
        hidden_states_35 = None
        hidden_states_37 = torch._C._nn.linear(
            hidden_states_36,
            l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_36 = l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_
        ) = None
        hidden_states_38 = torch.nn.functional.dropout(
            hidden_states_37, p=0.1, training=False
        )
        hidden_states_37 = None
        hidden_states_39 = hidden_states_34 + hidden_states_38
        hidden_states_34 = hidden_states_38 = None
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (256,),
            l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_39 = l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_41 = hidden_states_40 + object_queries
        linear_24 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_8 = linear_24 * 0.1767766952966369
        linear_24 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_41 = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_36 = linear_25.view(1, -1, 8, 32)
        linear_25 = None
        transpose_20 = view_36.transpose(1, 2)
        view_36 = None
        key_states_8 = transpose_20.contiguous()
        transpose_20 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_37 = linear_26.view(1, -1, 8, 32)
        linear_26 = None
        transpose_21 = view_37.transpose(1, 2)
        view_37 = None
        value_states_8 = transpose_21.contiguous()
        transpose_21 = None
        view_38 = query_states_8.view(1, 625, 8, 32)
        query_states_8 = None
        transpose_22 = view_38.transpose(1, 2)
        view_38 = None
        contiguous_14 = transpose_22.contiguous()
        transpose_22 = None
        query_states_9 = contiguous_14.view(8, -1, 32)
        contiguous_14 = None
        key_states_9 = key_states_8.view(8, -1, 32)
        key_states_8 = None
        value_states_9 = value_states_8.view(8, -1, 32)
        value_states_8 = None
        transpose_23 = key_states_9.transpose(1, 2)
        key_states_9 = None
        attn_weights_16 = torch.bmm(query_states_9, transpose_23)
        query_states_9 = transpose_23 = None
        view_42 = attn_weights_16.view(1, 8, 625, 625)
        attn_weights_16 = None
        attn_weights_17 = view_42 + attention_mask
        view_42 = None
        attn_weights_18 = attn_weights_17.view(8, 625, 625)
        attn_weights_17 = None
        attn_weights_19 = torch.nn.functional.softmax(attn_weights_18, dim=-1)
        attn_weights_18 = None
        attn_probs_4 = torch.nn.functional.dropout(
            attn_weights_19, p=0.0, training=False
        )
        attn_weights_19 = None
        attn_output_20 = torch.bmm(attn_probs_4, value_states_9)
        attn_probs_4 = value_states_9 = None
        attn_output_21 = attn_output_20.view(1, 8, 625, 32)
        attn_output_20 = None
        attn_output_22 = attn_output_21.transpose(1, 2)
        attn_output_21 = None
        attn_output_23 = attn_output_22.reshape(1, 625, 256)
        attn_output_22 = None
        attn_output_24 = torch._C._nn.linear(
            attn_output_23,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_23 = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_42 = torch.nn.functional.dropout(
            attn_output_24, p=0.1, training=False
        )
        attn_output_24 = None
        hidden_states_43 = hidden_states_40 + hidden_states_42
        hidden_states_40 = hidden_states_42 = None
        hidden_states_44 = torch.nn.functional.layer_norm(
            hidden_states_43,
            (256,),
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_43 = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_28 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_
        ) = None
        hidden_states_45 = torch.nn.functional.relu(linear_28, inplace=False)
        linear_28 = None
        hidden_states_46 = torch.nn.functional.dropout(
            hidden_states_45, p=0.0, training=False
        )
        hidden_states_45 = None
        hidden_states_47 = torch._C._nn.linear(
            hidden_states_46,
            l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_46 = l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_
        ) = None
        hidden_states_48 = torch.nn.functional.dropout(
            hidden_states_47, p=0.1, training=False
        )
        hidden_states_47 = None
        hidden_states_49 = hidden_states_44 + hidden_states_48
        hidden_states_44 = hidden_states_48 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (256,),
            l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_49 = l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_51 = hidden_states_50 + object_queries
        linear_30 = torch._C._nn.linear(
            hidden_states_51,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_10 = linear_30 * 0.1767766952966369
        linear_30 = None
        linear_31 = torch._C._nn.linear(
            hidden_states_51,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_51 = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_45 = linear_31.view(1, -1, 8, 32)
        linear_31 = None
        transpose_25 = view_45.transpose(1, 2)
        view_45 = None
        key_states_10 = transpose_25.contiguous()
        transpose_25 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_46 = linear_32.view(1, -1, 8, 32)
        linear_32 = None
        transpose_26 = view_46.transpose(1, 2)
        view_46 = None
        value_states_10 = transpose_26.contiguous()
        transpose_26 = None
        view_47 = query_states_10.view(1, 625, 8, 32)
        query_states_10 = None
        transpose_27 = view_47.transpose(1, 2)
        view_47 = None
        contiguous_17 = transpose_27.contiguous()
        transpose_27 = None
        query_states_11 = contiguous_17.view(8, -1, 32)
        contiguous_17 = None
        key_states_11 = key_states_10.view(8, -1, 32)
        key_states_10 = None
        value_states_11 = value_states_10.view(8, -1, 32)
        value_states_10 = None
        transpose_28 = key_states_11.transpose(1, 2)
        key_states_11 = None
        attn_weights_20 = torch.bmm(query_states_11, transpose_28)
        query_states_11 = transpose_28 = None
        view_51 = attn_weights_20.view(1, 8, 625, 625)
        attn_weights_20 = None
        attn_weights_21 = view_51 + attention_mask
        view_51 = attention_mask = None
        attn_weights_22 = attn_weights_21.view(8, 625, 625)
        attn_weights_21 = None
        attn_weights_23 = torch.nn.functional.softmax(attn_weights_22, dim=-1)
        attn_weights_22 = None
        attn_probs_5 = torch.nn.functional.dropout(
            attn_weights_23, p=0.0, training=False
        )
        attn_weights_23 = None
        attn_output_25 = torch.bmm(attn_probs_5, value_states_11)
        attn_probs_5 = value_states_11 = None
        attn_output_26 = attn_output_25.view(1, 8, 625, 32)
        attn_output_25 = None
        attn_output_27 = attn_output_26.transpose(1, 2)
        attn_output_26 = None
        attn_output_28 = attn_output_27.reshape(1, 625, 256)
        attn_output_27 = None
        attn_output_29 = torch._C._nn.linear(
            attn_output_28,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_28 = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_52 = torch.nn.functional.dropout(
            attn_output_29, p=0.1, training=False
        )
        attn_output_29 = None
        hidden_states_53 = hidden_states_50 + hidden_states_52
        hidden_states_50 = hidden_states_52 = None
        hidden_states_54 = torch.nn.functional.layer_norm(
            hidden_states_53,
            (256,),
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_53 = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        linear_34 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        ) = None
        hidden_states_55 = torch.nn.functional.relu(linear_34, inplace=False)
        linear_34 = None
        hidden_states_56 = torch.nn.functional.dropout(
            hidden_states_55, p=0.0, training=False
        )
        hidden_states_55 = None
        hidden_states_57 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_56 = l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = (
            l_self_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        ) = None
        hidden_states_58 = torch.nn.functional.dropout(
            hidden_states_57, p=0.1, training=False
        )
        hidden_states_57 = None
        hidden_states_59 = hidden_states_54 + hidden_states_58
        hidden_states_54 = hidden_states_58 = None
        hidden_states_60 = torch.nn.functional.layer_norm(
            hidden_states_59,
            (256,),
            l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_59 = l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = (None)
        unsqueeze = (
            l_self_modules_query_position_embeddings_parameters_weight_.unsqueeze(0)
        )
        l_self_modules_query_position_embeddings_parameters_weight_ = None
        query_position_embeddings = unsqueeze.repeat(1, 1, 1)
        unsqueeze = None
        queries = torch.zeros_like(query_position_embeddings)
        getitem_41 = flattened_mask[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        flattened_mask = None
        expand_1 = getitem_41.expand(1, 1, 15, 625)
        getitem_41 = None
        expanded_mask_1 = expand_1.to(torch.float32)
        expand_1 = None
        tensor_1 = torch.tensor(1.0, dtype=torch.float32)
        inverted_mask_1 = tensor_1 - expanded_mask_1
        tensor_1 = expanded_mask_1 = None
        to_11 = inverted_mask_1.to(torch.bool)
        encoder_attention_mask = inverted_mask_1.masked_fill(
            to_11, -3.4028234663852886e38
        )
        inverted_mask_1 = to_11 = None
        hidden_states_61 = queries + query_position_embeddings
        linear_36 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_12 = linear_36 * 0.1767766952966369
        linear_36 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_61 = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_54 = linear_37.view(1, -1, 8, 32)
        linear_37 = None
        transpose_30 = view_54.transpose(1, 2)
        view_54 = None
        key_states_12 = transpose_30.contiguous()
        transpose_30 = None
        linear_38 = torch._C._nn.linear(
            queries,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_55 = linear_38.view(1, -1, 8, 32)
        linear_38 = None
        transpose_31 = view_55.transpose(1, 2)
        view_55 = None
        value_states_12 = transpose_31.contiguous()
        transpose_31 = None
        view_56 = query_states_12.view(1, 15, 8, 32)
        query_states_12 = None
        transpose_32 = view_56.transpose(1, 2)
        view_56 = None
        contiguous_20 = transpose_32.contiguous()
        transpose_32 = None
        query_states_13 = contiguous_20.view(8, -1, 32)
        contiguous_20 = None
        key_states_13 = key_states_12.view(8, -1, 32)
        key_states_12 = None
        value_states_13 = value_states_12.view(8, -1, 32)
        value_states_12 = None
        transpose_33 = key_states_13.transpose(1, 2)
        key_states_13 = None
        attn_weights_24 = torch.bmm(query_states_13, transpose_33)
        query_states_13 = transpose_33 = None
        attn_weights_25 = torch.nn.functional.softmax(attn_weights_24, dim=-1)
        attn_weights_24 = None
        attn_probs_6 = torch.nn.functional.dropout(
            attn_weights_25, p=0.0, training=False
        )
        attn_weights_25 = None
        attn_output_30 = torch.bmm(attn_probs_6, value_states_13)
        attn_probs_6 = value_states_13 = None
        attn_output_31 = attn_output_30.view(1, 8, 15, 32)
        attn_output_30 = None
        attn_output_32 = attn_output_31.transpose(1, 2)
        attn_output_31 = None
        attn_output_33 = attn_output_32.reshape(1, 15, 256)
        attn_output_32 = None
        attn_output_34 = torch._C._nn.linear(
            attn_output_33,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_33 = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_62 = torch.nn.functional.dropout(
            attn_output_34, p=0.1, training=False
        )
        attn_output_34 = None
        hidden_states_63 = queries + hidden_states_62
        queries = hidden_states_62 = None
        hidden_states_64 = torch.nn.functional.layer_norm(
            hidden_states_63,
            (256,),
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_63 = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_65 = hidden_states_64 + query_position_embeddings
        key_value_states = hidden_states_60 + object_queries
        linear_40 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_65 = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_14 = linear_40 * 0.1767766952966369
        linear_40 = None
        linear_41 = torch._C._nn.linear(
            key_value_states,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        key_value_states = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        view_61 = linear_41.view(1, -1, 8, 32)
        linear_41 = None
        transpose_35 = view_61.transpose(1, 2)
        view_61 = None
        key_states_14 = transpose_35.contiguous()
        transpose_35 = None
        linear_42 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_62 = linear_42.view(1, -1, 8, 32)
        linear_42 = None
        transpose_36 = view_62.transpose(1, 2)
        view_62 = None
        value_states_14 = transpose_36.contiguous()
        transpose_36 = None
        view_63 = query_states_14.view(1, 15, 8, 32)
        query_states_14 = None
        transpose_37 = view_63.transpose(1, 2)
        view_63 = None
        contiguous_23 = transpose_37.contiguous()
        transpose_37 = None
        query_states_15 = contiguous_23.view(8, -1, 32)
        contiguous_23 = None
        key_states_15 = key_states_14.view(8, -1, 32)
        key_states_14 = None
        value_states_15 = value_states_14.view(8, -1, 32)
        value_states_14 = None
        transpose_38 = key_states_15.transpose(1, 2)
        key_states_15 = None
        attn_weights_26 = torch.bmm(query_states_15, transpose_38)
        query_states_15 = transpose_38 = None
        view_67 = attn_weights_26.view(1, 8, 15, 625)
        attn_weights_26 = None
        attn_weights_27 = view_67 + encoder_attention_mask
        view_67 = None
        attn_weights_28 = attn_weights_27.view(8, 15, 625)
        attn_weights_27 = None
        attn_weights_29 = torch.nn.functional.softmax(attn_weights_28, dim=-1)
        attn_weights_28 = None
        attn_probs_7 = torch.nn.functional.dropout(
            attn_weights_29, p=0.0, training=False
        )
        attn_weights_29 = None
        attn_output_35 = torch.bmm(attn_probs_7, value_states_15)
        attn_probs_7 = value_states_15 = None
        attn_output_36 = attn_output_35.view(1, 8, 15, 32)
        attn_output_35 = None
        attn_output_37 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_38 = attn_output_37.reshape(1, 15, 256)
        attn_output_37 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_38 = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_66 = torch.nn.functional.dropout(
            attn_output_39, p=0.1, training=False
        )
        attn_output_39 = None
        hidden_states_67 = hidden_states_64 + hidden_states_66
        hidden_states_64 = hidden_states_66 = None
        hidden_states_68 = torch.nn.functional.layer_norm(
            hidden_states_67,
            (256,),
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_67 = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_44 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_
        ) = None
        hidden_states_69 = torch.nn.functional.relu(linear_44, inplace=False)
        linear_44 = None
        hidden_states_70 = torch.nn.functional.dropout(
            hidden_states_69, p=0.0, training=False
        )
        hidden_states_69 = None
        hidden_states_71 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_,
        )
        hidden_states_70 = l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_
        ) = None
        hidden_states_72 = torch.nn.functional.dropout(
            hidden_states_71, p=0.1, training=False
        )
        hidden_states_71 = None
        hidden_states_73 = hidden_states_68 + hidden_states_72
        hidden_states_68 = hidden_states_72 = None
        hidden_states_74 = torch.nn.functional.layer_norm(
            hidden_states_73,
            (256,),
            l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_73 = l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_75 = hidden_states_74 + query_position_embeddings
        linear_46 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_16 = linear_46 * 0.1767766952966369
        linear_46 = None
        linear_47 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_75 = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_70 = linear_47.view(1, -1, 8, 32)
        linear_47 = None
        transpose_40 = view_70.transpose(1, 2)
        view_70 = None
        key_states_16 = transpose_40.contiguous()
        transpose_40 = None
        linear_48 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_71 = linear_48.view(1, -1, 8, 32)
        linear_48 = None
        transpose_41 = view_71.transpose(1, 2)
        view_71 = None
        value_states_16 = transpose_41.contiguous()
        transpose_41 = None
        view_72 = query_states_16.view(1, 15, 8, 32)
        query_states_16 = None
        transpose_42 = view_72.transpose(1, 2)
        view_72 = None
        contiguous_26 = transpose_42.contiguous()
        transpose_42 = None
        query_states_17 = contiguous_26.view(8, -1, 32)
        contiguous_26 = None
        key_states_17 = key_states_16.view(8, -1, 32)
        key_states_16 = None
        value_states_17 = value_states_16.view(8, -1, 32)
        value_states_16 = None
        transpose_43 = key_states_17.transpose(1, 2)
        key_states_17 = None
        attn_weights_30 = torch.bmm(query_states_17, transpose_43)
        query_states_17 = transpose_43 = None
        attn_weights_31 = torch.nn.functional.softmax(attn_weights_30, dim=-1)
        attn_weights_30 = None
        attn_probs_8 = torch.nn.functional.dropout(
            attn_weights_31, p=0.0, training=False
        )
        attn_weights_31 = None
        attn_output_40 = torch.bmm(attn_probs_8, value_states_17)
        attn_probs_8 = value_states_17 = None
        attn_output_41 = attn_output_40.view(1, 8, 15, 32)
        attn_output_40 = None
        attn_output_42 = attn_output_41.transpose(1, 2)
        attn_output_41 = None
        attn_output_43 = attn_output_42.reshape(1, 15, 256)
        attn_output_42 = None
        attn_output_44 = torch._C._nn.linear(
            attn_output_43,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_43 = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_76 = torch.nn.functional.dropout(
            attn_output_44, p=0.1, training=False
        )
        attn_output_44 = None
        hidden_states_77 = hidden_states_74 + hidden_states_76
        hidden_states_74 = hidden_states_76 = None
        hidden_states_78 = torch.nn.functional.layer_norm(
            hidden_states_77,
            (256,),
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_77 = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_79 = hidden_states_78 + query_position_embeddings
        key_value_states_1 = hidden_states_60 + object_queries
        linear_50 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_79 = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_18 = linear_50 * 0.1767766952966369
        linear_50 = None
        linear_51 = torch._C._nn.linear(
            key_value_states_1,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        key_value_states_1 = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        view_77 = linear_51.view(1, -1, 8, 32)
        linear_51 = None
        transpose_45 = view_77.transpose(1, 2)
        view_77 = None
        key_states_18 = transpose_45.contiguous()
        transpose_45 = None
        linear_52 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_78 = linear_52.view(1, -1, 8, 32)
        linear_52 = None
        transpose_46 = view_78.transpose(1, 2)
        view_78 = None
        value_states_18 = transpose_46.contiguous()
        transpose_46 = None
        view_79 = query_states_18.view(1, 15, 8, 32)
        query_states_18 = None
        transpose_47 = view_79.transpose(1, 2)
        view_79 = None
        contiguous_29 = transpose_47.contiguous()
        transpose_47 = None
        query_states_19 = contiguous_29.view(8, -1, 32)
        contiguous_29 = None
        key_states_19 = key_states_18.view(8, -1, 32)
        key_states_18 = None
        value_states_19 = value_states_18.view(8, -1, 32)
        value_states_18 = None
        transpose_48 = key_states_19.transpose(1, 2)
        key_states_19 = None
        attn_weights_32 = torch.bmm(query_states_19, transpose_48)
        query_states_19 = transpose_48 = None
        view_83 = attn_weights_32.view(1, 8, 15, 625)
        attn_weights_32 = None
        attn_weights_33 = view_83 + encoder_attention_mask
        view_83 = None
        attn_weights_34 = attn_weights_33.view(8, 15, 625)
        attn_weights_33 = None
        attn_weights_35 = torch.nn.functional.softmax(attn_weights_34, dim=-1)
        attn_weights_34 = None
        attn_probs_9 = torch.nn.functional.dropout(
            attn_weights_35, p=0.0, training=False
        )
        attn_weights_35 = None
        attn_output_45 = torch.bmm(attn_probs_9, value_states_19)
        attn_probs_9 = value_states_19 = None
        attn_output_46 = attn_output_45.view(1, 8, 15, 32)
        attn_output_45 = None
        attn_output_47 = attn_output_46.transpose(1, 2)
        attn_output_46 = None
        attn_output_48 = attn_output_47.reshape(1, 15, 256)
        attn_output_47 = None
        attn_output_49 = torch._C._nn.linear(
            attn_output_48,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_48 = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_80 = torch.nn.functional.dropout(
            attn_output_49, p=0.1, training=False
        )
        attn_output_49 = None
        hidden_states_81 = hidden_states_78 + hidden_states_80
        hidden_states_78 = hidden_states_80 = None
        hidden_states_82 = torch.nn.functional.layer_norm(
            hidden_states_81,
            (256,),
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_81 = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_54 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_
        ) = None
        hidden_states_83 = torch.nn.functional.relu(linear_54, inplace=False)
        linear_54 = None
        hidden_states_84 = torch.nn.functional.dropout(
            hidden_states_83, p=0.0, training=False
        )
        hidden_states_83 = None
        hidden_states_85 = torch._C._nn.linear(
            hidden_states_84,
            l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_,
        )
        hidden_states_84 = l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_
        ) = None
        hidden_states_86 = torch.nn.functional.dropout(
            hidden_states_85, p=0.1, training=False
        )
        hidden_states_85 = None
        hidden_states_87 = hidden_states_82 + hidden_states_86
        hidden_states_82 = hidden_states_86 = None
        hidden_states_88 = torch.nn.functional.layer_norm(
            hidden_states_87,
            (256,),
            l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_87 = l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_89 = hidden_states_88 + query_position_embeddings
        linear_56 = torch._C._nn.linear(
            hidden_states_89,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_20 = linear_56 * 0.1767766952966369
        linear_56 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_89,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_89 = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_86 = linear_57.view(1, -1, 8, 32)
        linear_57 = None
        transpose_50 = view_86.transpose(1, 2)
        view_86 = None
        key_states_20 = transpose_50.contiguous()
        transpose_50 = None
        linear_58 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_87 = linear_58.view(1, -1, 8, 32)
        linear_58 = None
        transpose_51 = view_87.transpose(1, 2)
        view_87 = None
        value_states_20 = transpose_51.contiguous()
        transpose_51 = None
        view_88 = query_states_20.view(1, 15, 8, 32)
        query_states_20 = None
        transpose_52 = view_88.transpose(1, 2)
        view_88 = None
        contiguous_32 = transpose_52.contiguous()
        transpose_52 = None
        query_states_21 = contiguous_32.view(8, -1, 32)
        contiguous_32 = None
        key_states_21 = key_states_20.view(8, -1, 32)
        key_states_20 = None
        value_states_21 = value_states_20.view(8, -1, 32)
        value_states_20 = None
        transpose_53 = key_states_21.transpose(1, 2)
        key_states_21 = None
        attn_weights_36 = torch.bmm(query_states_21, transpose_53)
        query_states_21 = transpose_53 = None
        attn_weights_37 = torch.nn.functional.softmax(attn_weights_36, dim=-1)
        attn_weights_36 = None
        attn_probs_10 = torch.nn.functional.dropout(
            attn_weights_37, p=0.0, training=False
        )
        attn_weights_37 = None
        attn_output_50 = torch.bmm(attn_probs_10, value_states_21)
        attn_probs_10 = value_states_21 = None
        attn_output_51 = attn_output_50.view(1, 8, 15, 32)
        attn_output_50 = None
        attn_output_52 = attn_output_51.transpose(1, 2)
        attn_output_51 = None
        attn_output_53 = attn_output_52.reshape(1, 15, 256)
        attn_output_52 = None
        attn_output_54 = torch._C._nn.linear(
            attn_output_53,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_53 = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_90 = torch.nn.functional.dropout(
            attn_output_54, p=0.1, training=False
        )
        attn_output_54 = None
        hidden_states_91 = hidden_states_88 + hidden_states_90
        hidden_states_88 = hidden_states_90 = None
        hidden_states_92 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (256,),
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_91 = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_93 = hidden_states_92 + query_position_embeddings
        key_value_states_2 = hidden_states_60 + object_queries
        linear_60 = torch._C._nn.linear(
            hidden_states_93,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_93 = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_22 = linear_60 * 0.1767766952966369
        linear_60 = None
        linear_61 = torch._C._nn.linear(
            key_value_states_2,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        key_value_states_2 = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        view_93 = linear_61.view(1, -1, 8, 32)
        linear_61 = None
        transpose_55 = view_93.transpose(1, 2)
        view_93 = None
        key_states_22 = transpose_55.contiguous()
        transpose_55 = None
        linear_62 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_94 = linear_62.view(1, -1, 8, 32)
        linear_62 = None
        transpose_56 = view_94.transpose(1, 2)
        view_94 = None
        value_states_22 = transpose_56.contiguous()
        transpose_56 = None
        view_95 = query_states_22.view(1, 15, 8, 32)
        query_states_22 = None
        transpose_57 = view_95.transpose(1, 2)
        view_95 = None
        contiguous_35 = transpose_57.contiguous()
        transpose_57 = None
        query_states_23 = contiguous_35.view(8, -1, 32)
        contiguous_35 = None
        key_states_23 = key_states_22.view(8, -1, 32)
        key_states_22 = None
        value_states_23 = value_states_22.view(8, -1, 32)
        value_states_22 = None
        transpose_58 = key_states_23.transpose(1, 2)
        key_states_23 = None
        attn_weights_38 = torch.bmm(query_states_23, transpose_58)
        query_states_23 = transpose_58 = None
        view_99 = attn_weights_38.view(1, 8, 15, 625)
        attn_weights_38 = None
        attn_weights_39 = view_99 + encoder_attention_mask
        view_99 = None
        attn_weights_40 = attn_weights_39.view(8, 15, 625)
        attn_weights_39 = None
        attn_weights_41 = torch.nn.functional.softmax(attn_weights_40, dim=-1)
        attn_weights_40 = None
        attn_probs_11 = torch.nn.functional.dropout(
            attn_weights_41, p=0.0, training=False
        )
        attn_weights_41 = None
        attn_output_55 = torch.bmm(attn_probs_11, value_states_23)
        attn_probs_11 = value_states_23 = None
        attn_output_56 = attn_output_55.view(1, 8, 15, 32)
        attn_output_55 = None
        attn_output_57 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_58 = attn_output_57.reshape(1, 15, 256)
        attn_output_57 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_58 = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_94 = torch.nn.functional.dropout(
            attn_output_59, p=0.1, training=False
        )
        attn_output_59 = None
        hidden_states_95 = hidden_states_92 + hidden_states_94
        hidden_states_92 = hidden_states_94 = None
        hidden_states_96 = torch.nn.functional.layer_norm(
            hidden_states_95,
            (256,),
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_95 = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_64 = torch._C._nn.linear(
            hidden_states_96,
            l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_
        ) = None
        hidden_states_97 = torch.nn.functional.relu(linear_64, inplace=False)
        linear_64 = None
        hidden_states_98 = torch.nn.functional.dropout(
            hidden_states_97, p=0.0, training=False
        )
        hidden_states_97 = None
        hidden_states_99 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_,
        )
        hidden_states_98 = l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_
        ) = None
        hidden_states_100 = torch.nn.functional.dropout(
            hidden_states_99, p=0.1, training=False
        )
        hidden_states_99 = None
        hidden_states_101 = hidden_states_96 + hidden_states_100
        hidden_states_96 = hidden_states_100 = None
        hidden_states_102 = torch.nn.functional.layer_norm(
            hidden_states_101,
            (256,),
            l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_101 = l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_103 = hidden_states_102 + query_position_embeddings
        linear_66 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_24 = linear_66 * 0.1767766952966369
        linear_66 = None
        linear_67 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_103 = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_102 = linear_67.view(1, -1, 8, 32)
        linear_67 = None
        transpose_60 = view_102.transpose(1, 2)
        view_102 = None
        key_states_24 = transpose_60.contiguous()
        transpose_60 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_103 = linear_68.view(1, -1, 8, 32)
        linear_68 = None
        transpose_61 = view_103.transpose(1, 2)
        view_103 = None
        value_states_24 = transpose_61.contiguous()
        transpose_61 = None
        view_104 = query_states_24.view(1, 15, 8, 32)
        query_states_24 = None
        transpose_62 = view_104.transpose(1, 2)
        view_104 = None
        contiguous_38 = transpose_62.contiguous()
        transpose_62 = None
        query_states_25 = contiguous_38.view(8, -1, 32)
        contiguous_38 = None
        key_states_25 = key_states_24.view(8, -1, 32)
        key_states_24 = None
        value_states_25 = value_states_24.view(8, -1, 32)
        value_states_24 = None
        transpose_63 = key_states_25.transpose(1, 2)
        key_states_25 = None
        attn_weights_42 = torch.bmm(query_states_25, transpose_63)
        query_states_25 = transpose_63 = None
        attn_weights_43 = torch.nn.functional.softmax(attn_weights_42, dim=-1)
        attn_weights_42 = None
        attn_probs_12 = torch.nn.functional.dropout(
            attn_weights_43, p=0.0, training=False
        )
        attn_weights_43 = None
        attn_output_60 = torch.bmm(attn_probs_12, value_states_25)
        attn_probs_12 = value_states_25 = None
        attn_output_61 = attn_output_60.view(1, 8, 15, 32)
        attn_output_60 = None
        attn_output_62 = attn_output_61.transpose(1, 2)
        attn_output_61 = None
        attn_output_63 = attn_output_62.reshape(1, 15, 256)
        attn_output_62 = None
        attn_output_64 = torch._C._nn.linear(
            attn_output_63,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_63 = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_104 = torch.nn.functional.dropout(
            attn_output_64, p=0.1, training=False
        )
        attn_output_64 = None
        hidden_states_105 = hidden_states_102 + hidden_states_104
        hidden_states_102 = hidden_states_104 = None
        hidden_states_106 = torch.nn.functional.layer_norm(
            hidden_states_105,
            (256,),
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_105 = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_107 = hidden_states_106 + query_position_embeddings
        key_value_states_3 = hidden_states_60 + object_queries
        linear_70 = torch._C._nn.linear(
            hidden_states_107,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_107 = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_26 = linear_70 * 0.1767766952966369
        linear_70 = None
        linear_71 = torch._C._nn.linear(
            key_value_states_3,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        key_value_states_3 = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        view_109 = linear_71.view(1, -1, 8, 32)
        linear_71 = None
        transpose_65 = view_109.transpose(1, 2)
        view_109 = None
        key_states_26 = transpose_65.contiguous()
        transpose_65 = None
        linear_72 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_110 = linear_72.view(1, -1, 8, 32)
        linear_72 = None
        transpose_66 = view_110.transpose(1, 2)
        view_110 = None
        value_states_26 = transpose_66.contiguous()
        transpose_66 = None
        view_111 = query_states_26.view(1, 15, 8, 32)
        query_states_26 = None
        transpose_67 = view_111.transpose(1, 2)
        view_111 = None
        contiguous_41 = transpose_67.contiguous()
        transpose_67 = None
        query_states_27 = contiguous_41.view(8, -1, 32)
        contiguous_41 = None
        key_states_27 = key_states_26.view(8, -1, 32)
        key_states_26 = None
        value_states_27 = value_states_26.view(8, -1, 32)
        value_states_26 = None
        transpose_68 = key_states_27.transpose(1, 2)
        key_states_27 = None
        attn_weights_44 = torch.bmm(query_states_27, transpose_68)
        query_states_27 = transpose_68 = None
        view_115 = attn_weights_44.view(1, 8, 15, 625)
        attn_weights_44 = None
        attn_weights_45 = view_115 + encoder_attention_mask
        view_115 = None
        attn_weights_46 = attn_weights_45.view(8, 15, 625)
        attn_weights_45 = None
        attn_weights_47 = torch.nn.functional.softmax(attn_weights_46, dim=-1)
        attn_weights_46 = None
        attn_probs_13 = torch.nn.functional.dropout(
            attn_weights_47, p=0.0, training=False
        )
        attn_weights_47 = None
        attn_output_65 = torch.bmm(attn_probs_13, value_states_27)
        attn_probs_13 = value_states_27 = None
        attn_output_66 = attn_output_65.view(1, 8, 15, 32)
        attn_output_65 = None
        attn_output_67 = attn_output_66.transpose(1, 2)
        attn_output_66 = None
        attn_output_68 = attn_output_67.reshape(1, 15, 256)
        attn_output_67 = None
        attn_output_69 = torch._C._nn.linear(
            attn_output_68,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_68 = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_108 = torch.nn.functional.dropout(
            attn_output_69, p=0.1, training=False
        )
        attn_output_69 = None
        hidden_states_109 = hidden_states_106 + hidden_states_108
        hidden_states_106 = hidden_states_108 = None
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (256,),
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_109 = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_74 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_
        ) = None
        hidden_states_111 = torch.nn.functional.relu(linear_74, inplace=False)
        linear_74 = None
        hidden_states_112 = torch.nn.functional.dropout(
            hidden_states_111, p=0.0, training=False
        )
        hidden_states_111 = None
        hidden_states_113 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_,
        )
        hidden_states_112 = l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_
        ) = None
        hidden_states_114 = torch.nn.functional.dropout(
            hidden_states_113, p=0.1, training=False
        )
        hidden_states_113 = None
        hidden_states_115 = hidden_states_110 + hidden_states_114
        hidden_states_110 = hidden_states_114 = None
        hidden_states_116 = torch.nn.functional.layer_norm(
            hidden_states_115,
            (256,),
            l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_115 = l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_117 = hidden_states_116 + query_position_embeddings
        linear_76 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_28 = linear_76 * 0.1767766952966369
        linear_76 = None
        linear_77 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_117 = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_118 = linear_77.view(1, -1, 8, 32)
        linear_77 = None
        transpose_70 = view_118.transpose(1, 2)
        view_118 = None
        key_states_28 = transpose_70.contiguous()
        transpose_70 = None
        linear_78 = torch._C._nn.linear(
            hidden_states_116,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_119 = linear_78.view(1, -1, 8, 32)
        linear_78 = None
        transpose_71 = view_119.transpose(1, 2)
        view_119 = None
        value_states_28 = transpose_71.contiguous()
        transpose_71 = None
        view_120 = query_states_28.view(1, 15, 8, 32)
        query_states_28 = None
        transpose_72 = view_120.transpose(1, 2)
        view_120 = None
        contiguous_44 = transpose_72.contiguous()
        transpose_72 = None
        query_states_29 = contiguous_44.view(8, -1, 32)
        contiguous_44 = None
        key_states_29 = key_states_28.view(8, -1, 32)
        key_states_28 = None
        value_states_29 = value_states_28.view(8, -1, 32)
        value_states_28 = None
        transpose_73 = key_states_29.transpose(1, 2)
        key_states_29 = None
        attn_weights_48 = torch.bmm(query_states_29, transpose_73)
        query_states_29 = transpose_73 = None
        attn_weights_49 = torch.nn.functional.softmax(attn_weights_48, dim=-1)
        attn_weights_48 = None
        attn_probs_14 = torch.nn.functional.dropout(
            attn_weights_49, p=0.0, training=False
        )
        attn_weights_49 = None
        attn_output_70 = torch.bmm(attn_probs_14, value_states_29)
        attn_probs_14 = value_states_29 = None
        attn_output_71 = attn_output_70.view(1, 8, 15, 32)
        attn_output_70 = None
        attn_output_72 = attn_output_71.transpose(1, 2)
        attn_output_71 = None
        attn_output_73 = attn_output_72.reshape(1, 15, 256)
        attn_output_72 = None
        attn_output_74 = torch._C._nn.linear(
            attn_output_73,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_73 = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_118 = torch.nn.functional.dropout(
            attn_output_74, p=0.1, training=False
        )
        attn_output_74 = None
        hidden_states_119 = hidden_states_116 + hidden_states_118
        hidden_states_116 = hidden_states_118 = None
        hidden_states_120 = torch.nn.functional.layer_norm(
            hidden_states_119,
            (256,),
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_119 = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_121 = hidden_states_120 + query_position_embeddings
        key_value_states_4 = hidden_states_60 + object_queries
        linear_80 = torch._C._nn.linear(
            hidden_states_121,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_121 = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_30 = linear_80 * 0.1767766952966369
        linear_80 = None
        linear_81 = torch._C._nn.linear(
            key_value_states_4,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        key_value_states_4 = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        view_125 = linear_81.view(1, -1, 8, 32)
        linear_81 = None
        transpose_75 = view_125.transpose(1, 2)
        view_125 = None
        key_states_30 = transpose_75.contiguous()
        transpose_75 = None
        linear_82 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_126 = linear_82.view(1, -1, 8, 32)
        linear_82 = None
        transpose_76 = view_126.transpose(1, 2)
        view_126 = None
        value_states_30 = transpose_76.contiguous()
        transpose_76 = None
        view_127 = query_states_30.view(1, 15, 8, 32)
        query_states_30 = None
        transpose_77 = view_127.transpose(1, 2)
        view_127 = None
        contiguous_47 = transpose_77.contiguous()
        transpose_77 = None
        query_states_31 = contiguous_47.view(8, -1, 32)
        contiguous_47 = None
        key_states_31 = key_states_30.view(8, -1, 32)
        key_states_30 = None
        value_states_31 = value_states_30.view(8, -1, 32)
        value_states_30 = None
        transpose_78 = key_states_31.transpose(1, 2)
        key_states_31 = None
        attn_weights_50 = torch.bmm(query_states_31, transpose_78)
        query_states_31 = transpose_78 = None
        view_131 = attn_weights_50.view(1, 8, 15, 625)
        attn_weights_50 = None
        attn_weights_51 = view_131 + encoder_attention_mask
        view_131 = None
        attn_weights_52 = attn_weights_51.view(8, 15, 625)
        attn_weights_51 = None
        attn_weights_53 = torch.nn.functional.softmax(attn_weights_52, dim=-1)
        attn_weights_52 = None
        attn_probs_15 = torch.nn.functional.dropout(
            attn_weights_53, p=0.0, training=False
        )
        attn_weights_53 = None
        attn_output_75 = torch.bmm(attn_probs_15, value_states_31)
        attn_probs_15 = value_states_31 = None
        attn_output_76 = attn_output_75.view(1, 8, 15, 32)
        attn_output_75 = None
        attn_output_77 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_78 = attn_output_77.reshape(1, 15, 256)
        attn_output_77 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_78 = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_122 = torch.nn.functional.dropout(
            attn_output_79, p=0.1, training=False
        )
        attn_output_79 = None
        hidden_states_123 = hidden_states_120 + hidden_states_122
        hidden_states_120 = hidden_states_122 = None
        hidden_states_124 = torch.nn.functional.layer_norm(
            hidden_states_123,
            (256,),
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_123 = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_84 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_
        ) = None
        hidden_states_125 = torch.nn.functional.relu(linear_84, inplace=False)
        linear_84 = None
        hidden_states_126 = torch.nn.functional.dropout(
            hidden_states_125, p=0.0, training=False
        )
        hidden_states_125 = None
        hidden_states_127 = torch._C._nn.linear(
            hidden_states_126,
            l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_,
        )
        hidden_states_126 = l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_
        ) = None
        hidden_states_128 = torch.nn.functional.dropout(
            hidden_states_127, p=0.1, training=False
        )
        hidden_states_127 = None
        hidden_states_129 = hidden_states_124 + hidden_states_128
        hidden_states_124 = hidden_states_128 = None
        hidden_states_130 = torch.nn.functional.layer_norm(
            hidden_states_129,
            (256,),
            l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_129 = l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_131 = hidden_states_130 + query_position_embeddings
        linear_86 = torch._C._nn.linear(
            hidden_states_131,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_32 = linear_86 * 0.1767766952966369
        linear_86 = None
        linear_87 = torch._C._nn.linear(
            hidden_states_131,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        hidden_states_131 = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_134 = linear_87.view(1, -1, 8, 32)
        linear_87 = None
        transpose_80 = view_134.transpose(1, 2)
        view_134 = None
        key_states_32 = transpose_80.contiguous()
        transpose_80 = None
        linear_88 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_135 = linear_88.view(1, -1, 8, 32)
        linear_88 = None
        transpose_81 = view_135.transpose(1, 2)
        view_135 = None
        value_states_32 = transpose_81.contiguous()
        transpose_81 = None
        view_136 = query_states_32.view(1, 15, 8, 32)
        query_states_32 = None
        transpose_82 = view_136.transpose(1, 2)
        view_136 = None
        contiguous_50 = transpose_82.contiguous()
        transpose_82 = None
        query_states_33 = contiguous_50.view(8, -1, 32)
        contiguous_50 = None
        key_states_33 = key_states_32.view(8, -1, 32)
        key_states_32 = None
        value_states_33 = value_states_32.view(8, -1, 32)
        value_states_32 = None
        transpose_83 = key_states_33.transpose(1, 2)
        key_states_33 = None
        attn_weights_54 = torch.bmm(query_states_33, transpose_83)
        query_states_33 = transpose_83 = None
        attn_weights_55 = torch.nn.functional.softmax(attn_weights_54, dim=-1)
        attn_weights_54 = None
        attn_probs_16 = torch.nn.functional.dropout(
            attn_weights_55, p=0.0, training=False
        )
        attn_weights_55 = None
        attn_output_80 = torch.bmm(attn_probs_16, value_states_33)
        attn_probs_16 = value_states_33 = None
        attn_output_81 = attn_output_80.view(1, 8, 15, 32)
        attn_output_80 = None
        attn_output_82 = attn_output_81.transpose(1, 2)
        attn_output_81 = None
        attn_output_83 = attn_output_82.reshape(1, 15, 256)
        attn_output_82 = None
        attn_output_84 = torch._C._nn.linear(
            attn_output_83,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_83 = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_132 = torch.nn.functional.dropout(
            attn_output_84, p=0.1, training=False
        )
        attn_output_84 = None
        hidden_states_133 = hidden_states_130 + hidden_states_132
        hidden_states_130 = hidden_states_132 = None
        hidden_states_134 = torch.nn.functional.layer_norm(
            hidden_states_133,
            (256,),
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_133 = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_ = (None)
        hidden_states_135 = hidden_states_134 + query_position_embeddings
        query_position_embeddings = None
        key_value_states_5 = hidden_states_60 + object_queries
        object_queries = None
        linear_90 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_,
        )
        hidden_states_135 = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_ = (None)
        query_states_34 = linear_90 * 0.1767766952966369
        linear_90 = None
        linear_91 = torch._C._nn.linear(
            key_value_states_5,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_,
        )
        key_value_states_5 = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_ = (None)
        view_141 = linear_91.view(1, -1, 8, 32)
        linear_91 = None
        transpose_85 = view_141.transpose(1, 2)
        view_141 = None
        key_states_34 = transpose_85.contiguous()
        transpose_85 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_ = (None)
        view_142 = linear_92.view(1, -1, 8, 32)
        linear_92 = None
        transpose_86 = view_142.transpose(1, 2)
        view_142 = None
        value_states_34 = transpose_86.contiguous()
        transpose_86 = None
        view_143 = query_states_34.view(1, 15, 8, 32)
        query_states_34 = None
        transpose_87 = view_143.transpose(1, 2)
        view_143 = None
        contiguous_53 = transpose_87.contiguous()
        transpose_87 = None
        query_states_35 = contiguous_53.view(8, -1, 32)
        contiguous_53 = None
        key_states_35 = key_states_34.view(8, -1, 32)
        key_states_34 = None
        value_states_35 = value_states_34.view(8, -1, 32)
        value_states_34 = None
        transpose_88 = key_states_35.transpose(1, 2)
        key_states_35 = None
        attn_weights_56 = torch.bmm(query_states_35, transpose_88)
        query_states_35 = transpose_88 = None
        view_147 = attn_weights_56.view(1, 8, 15, 625)
        attn_weights_56 = None
        attn_weights_57 = view_147 + encoder_attention_mask
        view_147 = encoder_attention_mask = None
        attn_weights_58 = attn_weights_57.view(8, 15, 625)
        attn_weights_57 = None
        attn_weights_59 = torch.nn.functional.softmax(attn_weights_58, dim=-1)
        attn_weights_58 = None
        attn_probs_17 = torch.nn.functional.dropout(
            attn_weights_59, p=0.0, training=False
        )
        attn_weights_59 = None
        attn_output_85 = torch.bmm(attn_probs_17, value_states_35)
        attn_probs_17 = value_states_35 = None
        attn_output_86 = attn_output_85.view(1, 8, 15, 32)
        attn_output_85 = None
        attn_output_87 = attn_output_86.transpose(1, 2)
        attn_output_86 = None
        attn_output_88 = attn_output_87.reshape(1, 15, 256)
        attn_output_87 = None
        attn_output_89 = torch._C._nn.linear(
            attn_output_88,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_,
        )
        attn_output_88 = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_ = (None)
        hidden_states_136 = torch.nn.functional.dropout(
            attn_output_89, p=0.1, training=False
        )
        attn_output_89 = None
        hidden_states_137 = hidden_states_134 + hidden_states_136
        hidden_states_134 = hidden_states_136 = None
        hidden_states_138 = torch.nn.functional.layer_norm(
            hidden_states_137,
            (256,),
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_137 = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_ = (None)
        linear_94 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_,
        )
        l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_
        ) = None
        hidden_states_139 = torch.nn.functional.relu(linear_94, inplace=False)
        linear_94 = None
        hidden_states_140 = torch.nn.functional.dropout(
            hidden_states_139, p=0.0, training=False
        )
        hidden_states_139 = None
        hidden_states_141 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_,
        )
        hidden_states_140 = l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_ = (
            l_self_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_
        ) = None
        hidden_states_142 = torch.nn.functional.dropout(
            hidden_states_141, p=0.1, training=False
        )
        hidden_states_141 = None
        hidden_states_143 = hidden_states_138 + hidden_states_142
        hidden_states_138 = hidden_states_142 = None
        hidden_states_144 = torch.nn.functional.layer_norm(
            hidden_states_143,
            (256,),
            l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_,
            l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_143 = l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_ = l_self_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_ = (None)
        hidden_states_145 = torch.nn.functional.layer_norm(
            hidden_states_144,
            (256,),
            l_self_modules_decoder_modules_layernorm_parameters_weight_,
            l_self_modules_decoder_modules_layernorm_parameters_bias_,
            1e-05,
        )
        hidden_states_144 = (
            l_self_modules_decoder_modules_layernorm_parameters_weight_
        ) = l_self_modules_decoder_modules_layernorm_parameters_bias_ = None
        return (hidden_states_145, hidden_states_60)
