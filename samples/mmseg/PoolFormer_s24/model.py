import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_backbone_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_network_modules_0_modules_0_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_0_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_1_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_1_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_2_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_2_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_3_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_3_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_backbone_modules_patch_embed_modules_proj_parameters_weight_ = (
            L_self_modules_backbone_modules_patch_embed_modules_proj_parameters_weight_
        )
        l_self_modules_backbone_modules_patch_embed_modules_proj_parameters_bias_ = (
            L_self_modules_backbone_modules_patch_embed_modules_proj_parameters_bias_
        )
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_network_modules_0_modules_0_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_0_modules_0_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_0_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_0_modules_0_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_1_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_0_modules_1_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_1_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_0_modules_1_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_2_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_0_modules_2_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_2_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_0_modules_2_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_3_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_0_modules_3_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_3_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_0_modules_3_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_norm0_parameters_weight_ = (
            L_self_modules_backbone_modules_norm0_parameters_weight_
        )
        l_self_modules_backbone_modules_norm0_parameters_bias_ = (
            L_self_modules_backbone_modules_norm0_parameters_bias_
        )
        l_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_norm2_parameters_weight_ = (
            L_self_modules_backbone_modules_norm2_parameters_weight_
        )
        l_self_modules_backbone_modules_norm2_parameters_bias_ = (
            L_self_modules_backbone_modules_norm2_parameters_bias_
        )
        l_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_norm4_parameters_weight_ = (
            L_self_modules_backbone_modules_norm4_parameters_weight_
        )
        l_self_modules_backbone_modules_norm4_parameters_bias_ = (
            L_self_modules_backbone_modules_norm4_parameters_bias_
        )
        l_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_weight_ = L_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_weight_
        l_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_bias_ = L_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_norm6_parameters_weight_ = (
            L_self_modules_backbone_modules_norm6_parameters_weight_
        )
        l_self_modules_backbone_modules_norm6_parameters_bias_ = (
            L_self_modules_backbone_modules_norm6_parameters_bias_
        )
        l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_ = L_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_
        l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_ = L_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_
        l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_ = L_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_
        l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_bias_ = L_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_bias_
        l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_ = L_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_
        l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_ = L_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_
        l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_ = L_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_
        l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_ = L_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_
        l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_ = L_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
        )
        x = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_patch_embed_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_patch_embed_modules_proj_parameters_bias_,
            (4, 4),
            (2, 2),
            (1, 1),
            1,
        )
        l_inputs_ = (
            l_self_modules_backbone_modules_patch_embed_modules_proj_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_patch_embed_modules_proj_parameters_bias_
        ) = None
        unsqueeze = l_self_modules_backbone_modules_network_modules_0_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_1 = unsqueeze.unsqueeze(-1)
        unsqueeze = None
        group_norm = torch.nn.functional.group_norm(
            x,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm1_parameters_bias_ = (None)
        avg_pool2d = torch._C._nn.avg_pool2d(group_norm, 3, 1, 1, False, False, None)
        sub = avg_pool2d - group_norm
        avg_pool2d = group_norm = None
        mul = unsqueeze_1 * sub
        unsqueeze_1 = sub = None
        x_1 = x + mul
        x = mul = None
        unsqueeze_2 = l_self_modules_backbone_modules_network_modules_0_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_3 = unsqueeze_2.unsqueeze(-1)
        unsqueeze_2 = None
        group_norm_1 = torch.nn.functional.group_norm(
            x_1,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_0_modules_norm2_parameters_bias_ = (None)
        x_2 = torch.conv2d(
            group_norm_1,
            l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_1 = l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_3 = torch._C._nn.gelu(x_2, approximate="none")
        x_2 = None
        x_4 = torch.nn.functional.dropout(x_3, 0.0, False, False)
        x_3 = None
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_6 = torch.nn.functional.dropout(x_5, 0.0, False, False)
        x_5 = None
        mul_1 = unsqueeze_3 * x_6
        unsqueeze_3 = x_6 = None
        x_7 = x_1 + mul_1
        x_1 = mul_1 = None
        unsqueeze_4 = l_self_modules_backbone_modules_network_modules_0_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_5 = unsqueeze_4.unsqueeze(-1)
        unsqueeze_4 = None
        group_norm_2 = torch.nn.functional.group_norm(
            x_7,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_1 = torch._C._nn.avg_pool2d(
            group_norm_2, 3, 1, 1, False, False, None
        )
        sub_1 = avg_pool2d_1 - group_norm_2
        avg_pool2d_1 = group_norm_2 = None
        mul_2 = unsqueeze_5 * sub_1
        unsqueeze_5 = sub_1 = None
        x_8 = x_7 + mul_2
        x_7 = mul_2 = None
        unsqueeze_6 = l_self_modules_backbone_modules_network_modules_0_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_7 = unsqueeze_6.unsqueeze(-1)
        unsqueeze_6 = None
        group_norm_3 = torch.nn.functional.group_norm(
            x_8,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_1_modules_norm2_parameters_bias_ = (None)
        x_9 = torch.conv2d(
            group_norm_3,
            l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_3 = l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_10 = torch._C._nn.gelu(x_9, approximate="none")
        x_9 = None
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        mul_3 = unsqueeze_7 * x_13
        unsqueeze_7 = x_13 = None
        x_14 = x_8 + mul_3
        x_8 = mul_3 = None
        unsqueeze_8 = l_self_modules_backbone_modules_network_modules_0_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_9 = unsqueeze_8.unsqueeze(-1)
        unsqueeze_8 = None
        group_norm_4 = torch.nn.functional.group_norm(
            x_14,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_2 = torch._C._nn.avg_pool2d(
            group_norm_4, 3, 1, 1, False, False, None
        )
        sub_2 = avg_pool2d_2 - group_norm_4
        avg_pool2d_2 = group_norm_4 = None
        mul_4 = unsqueeze_9 * sub_2
        unsqueeze_9 = sub_2 = None
        x_15 = x_14 + mul_4
        x_14 = mul_4 = None
        unsqueeze_10 = l_self_modules_backbone_modules_network_modules_0_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_11 = unsqueeze_10.unsqueeze(-1)
        unsqueeze_10 = None
        group_norm_5 = torch.nn.functional.group_norm(
            x_15,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_2_modules_norm2_parameters_bias_ = (None)
        x_16 = torch.conv2d(
            group_norm_5,
            l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_5 = l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_17 = torch._C._nn.gelu(x_16, approximate="none")
        x_16 = None
        x_18 = torch.nn.functional.dropout(x_17, 0.0, False, False)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_20 = torch.nn.functional.dropout(x_19, 0.0, False, False)
        x_19 = None
        mul_5 = unsqueeze_11 * x_20
        unsqueeze_11 = x_20 = None
        x_21 = x_15 + mul_5
        x_15 = mul_5 = None
        unsqueeze_12 = l_self_modules_backbone_modules_network_modules_0_modules_3_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_3_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_13 = unsqueeze_12.unsqueeze(-1)
        unsqueeze_12 = None
        group_norm_6 = torch.nn.functional.group_norm(
            x_21,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_3 = torch._C._nn.avg_pool2d(
            group_norm_6, 3, 1, 1, False, False, None
        )
        sub_3 = avg_pool2d_3 - group_norm_6
        avg_pool2d_3 = group_norm_6 = None
        mul_6 = unsqueeze_13 * sub_3
        unsqueeze_13 = sub_3 = None
        x_22 = x_21 + mul_6
        x_21 = mul_6 = None
        unsqueeze_14 = l_self_modules_backbone_modules_network_modules_0_modules_3_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_3_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_15 = unsqueeze_14.unsqueeze(-1)
        unsqueeze_14 = None
        group_norm_7 = torch.nn.functional.group_norm(
            x_22,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_3_modules_norm2_parameters_bias_ = (None)
        x_23 = torch.conv2d(
            group_norm_7,
            l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_7 = l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_24 = torch._C._nn.gelu(x_23, approximate="none")
        x_23 = None
        x_25 = torch.nn.functional.dropout(x_24, 0.0, False, False)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_27 = torch.nn.functional.dropout(x_26, 0.0, False, False)
        x_26 = None
        mul_7 = unsqueeze_15 * x_27
        unsqueeze_15 = x_27 = None
        x_28 = x_22 + mul_7
        x_22 = mul_7 = None
        x_out = torch.nn.functional.group_norm(
            x_28,
            1,
            l_self_modules_backbone_modules_norm0_parameters_weight_,
            l_self_modules_backbone_modules_norm0_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_norm0_parameters_weight_ = (
            l_self_modules_backbone_modules_norm0_parameters_bias_
        ) = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_bias_ = (None)
        unsqueeze_16 = l_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_17 = unsqueeze_16.unsqueeze(-1)
        unsqueeze_16 = None
        group_norm_9 = torch.nn.functional.group_norm(
            x_29,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_4 = torch._C._nn.avg_pool2d(
            group_norm_9, 3, 1, 1, False, False, None
        )
        sub_4 = avg_pool2d_4 - group_norm_9
        avg_pool2d_4 = group_norm_9 = None
        mul_8 = unsqueeze_17 * sub_4
        unsqueeze_17 = sub_4 = None
        x_30 = x_29 + mul_8
        x_29 = mul_8 = None
        unsqueeze_18 = l_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_19 = unsqueeze_18.unsqueeze(-1)
        unsqueeze_18 = None
        group_norm_10 = torch.nn.functional.group_norm(
            x_30,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_bias_ = (None)
        x_31 = torch.conv2d(
            group_norm_10,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_10 = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_32 = torch._C._nn.gelu(x_31, approximate="none")
        x_31 = None
        x_33 = torch.nn.functional.dropout(x_32, 0.0, False, False)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_35 = torch.nn.functional.dropout(x_34, 0.0, False, False)
        x_34 = None
        mul_9 = unsqueeze_19 * x_35
        unsqueeze_19 = x_35 = None
        x_36 = x_30 + mul_9
        x_30 = mul_9 = None
        unsqueeze_20 = l_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_21 = unsqueeze_20.unsqueeze(-1)
        unsqueeze_20 = None
        group_norm_11 = torch.nn.functional.group_norm(
            x_36,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_5 = torch._C._nn.avg_pool2d(
            group_norm_11, 3, 1, 1, False, False, None
        )
        sub_5 = avg_pool2d_5 - group_norm_11
        avg_pool2d_5 = group_norm_11 = None
        mul_10 = unsqueeze_21 * sub_5
        unsqueeze_21 = sub_5 = None
        x_37 = x_36 + mul_10
        x_36 = mul_10 = None
        unsqueeze_22 = l_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_23 = unsqueeze_22.unsqueeze(-1)
        unsqueeze_22 = None
        group_norm_12 = torch.nn.functional.group_norm(
            x_37,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_bias_ = (None)
        x_38 = torch.conv2d(
            group_norm_12,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_12 = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_39 = torch._C._nn.gelu(x_38, approximate="none")
        x_38 = None
        x_40 = torch.nn.functional.dropout(x_39, 0.0, False, False)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_42 = torch.nn.functional.dropout(x_41, 0.0, False, False)
        x_41 = None
        mul_11 = unsqueeze_23 * x_42
        unsqueeze_23 = x_42 = None
        x_43 = x_37 + mul_11
        x_37 = mul_11 = None
        unsqueeze_24 = l_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_25 = unsqueeze_24.unsqueeze(-1)
        unsqueeze_24 = None
        group_norm_13 = torch.nn.functional.group_norm(
            x_43,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_6 = torch._C._nn.avg_pool2d(
            group_norm_13, 3, 1, 1, False, False, None
        )
        sub_6 = avg_pool2d_6 - group_norm_13
        avg_pool2d_6 = group_norm_13 = None
        mul_12 = unsqueeze_25 * sub_6
        unsqueeze_25 = sub_6 = None
        x_44 = x_43 + mul_12
        x_43 = mul_12 = None
        unsqueeze_26 = l_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_27 = unsqueeze_26.unsqueeze(-1)
        unsqueeze_26 = None
        group_norm_14 = torch.nn.functional.group_norm(
            x_44,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_bias_ = (None)
        x_45 = torch.conv2d(
            group_norm_14,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_14 = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_46 = torch._C._nn.gelu(x_45, approximate="none")
        x_45 = None
        x_47 = torch.nn.functional.dropout(x_46, 0.0, False, False)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_49 = torch.nn.functional.dropout(x_48, 0.0, False, False)
        x_48 = None
        mul_13 = unsqueeze_27 * x_49
        unsqueeze_27 = x_49 = None
        x_50 = x_44 + mul_13
        x_44 = mul_13 = None
        unsqueeze_28 = l_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_29 = unsqueeze_28.unsqueeze(-1)
        unsqueeze_28 = None
        group_norm_15 = torch.nn.functional.group_norm(
            x_50,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_7 = torch._C._nn.avg_pool2d(
            group_norm_15, 3, 1, 1, False, False, None
        )
        sub_7 = avg_pool2d_7 - group_norm_15
        avg_pool2d_7 = group_norm_15 = None
        mul_14 = unsqueeze_29 * sub_7
        unsqueeze_29 = sub_7 = None
        x_51 = x_50 + mul_14
        x_50 = mul_14 = None
        unsqueeze_30 = l_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_31 = unsqueeze_30.unsqueeze(-1)
        unsqueeze_30 = None
        group_norm_16 = torch.nn.functional.group_norm(
            x_51,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_bias_ = (None)
        x_52 = torch.conv2d(
            group_norm_16,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_16 = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_53 = torch._C._nn.gelu(x_52, approximate="none")
        x_52 = None
        x_54 = torch.nn.functional.dropout(x_53, 0.0, False, False)
        x_53 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_56 = torch.nn.functional.dropout(x_55, 0.0, False, False)
        x_55 = None
        mul_15 = unsqueeze_31 * x_56
        unsqueeze_31 = x_56 = None
        x_57 = x_51 + mul_15
        x_51 = mul_15 = None
        x_out_1 = torch.nn.functional.group_norm(
            x_57,
            1,
            l_self_modules_backbone_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_norm2_parameters_weight_ = (
            l_self_modules_backbone_modules_norm2_parameters_bias_
        ) = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_bias_ = (None)
        unsqueeze_32 = l_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_33 = unsqueeze_32.unsqueeze(-1)
        unsqueeze_32 = None
        group_norm_18 = torch.nn.functional.group_norm(
            x_58,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_8 = torch._C._nn.avg_pool2d(
            group_norm_18, 3, 1, 1, False, False, None
        )
        sub_8 = avg_pool2d_8 - group_norm_18
        avg_pool2d_8 = group_norm_18 = None
        mul_16 = unsqueeze_33 * sub_8
        unsqueeze_33 = sub_8 = None
        x_59 = x_58 + mul_16
        x_58 = mul_16 = None
        unsqueeze_34 = l_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_35 = unsqueeze_34.unsqueeze(-1)
        unsqueeze_34 = None
        group_norm_19 = torch.nn.functional.group_norm(
            x_59,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_bias_ = (None)
        x_60 = torch.conv2d(
            group_norm_19,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_19 = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_61 = torch._C._nn.gelu(x_60, approximate="none")
        x_60 = None
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_64 = torch.nn.functional.dropout(x_63, 0.0, False, False)
        x_63 = None
        mul_17 = unsqueeze_35 * x_64
        unsqueeze_35 = x_64 = None
        x_65 = x_59 + mul_17
        x_59 = mul_17 = None
        unsqueeze_36 = l_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_37 = unsqueeze_36.unsqueeze(-1)
        unsqueeze_36 = None
        group_norm_20 = torch.nn.functional.group_norm(
            x_65,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_9 = torch._C._nn.avg_pool2d(
            group_norm_20, 3, 1, 1, False, False, None
        )
        sub_9 = avg_pool2d_9 - group_norm_20
        avg_pool2d_9 = group_norm_20 = None
        mul_18 = unsqueeze_37 * sub_9
        unsqueeze_37 = sub_9 = None
        x_66 = x_65 + mul_18
        x_65 = mul_18 = None
        unsqueeze_38 = l_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_39 = unsqueeze_38.unsqueeze(-1)
        unsqueeze_38 = None
        group_norm_21 = torch.nn.functional.group_norm(
            x_66,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_bias_ = (None)
        x_67 = torch.conv2d(
            group_norm_21,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_21 = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_68 = torch._C._nn.gelu(x_67, approximate="none")
        x_67 = None
        x_69 = torch.nn.functional.dropout(x_68, 0.0, False, False)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_71 = torch.nn.functional.dropout(x_70, 0.0, False, False)
        x_70 = None
        mul_19 = unsqueeze_39 * x_71
        unsqueeze_39 = x_71 = None
        x_72 = x_66 + mul_19
        x_66 = mul_19 = None
        unsqueeze_40 = l_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_41 = unsqueeze_40.unsqueeze(-1)
        unsqueeze_40 = None
        group_norm_22 = torch.nn.functional.group_norm(
            x_72,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_10 = torch._C._nn.avg_pool2d(
            group_norm_22, 3, 1, 1, False, False, None
        )
        sub_10 = avg_pool2d_10 - group_norm_22
        avg_pool2d_10 = group_norm_22 = None
        mul_20 = unsqueeze_41 * sub_10
        unsqueeze_41 = sub_10 = None
        x_73 = x_72 + mul_20
        x_72 = mul_20 = None
        unsqueeze_42 = l_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_43 = unsqueeze_42.unsqueeze(-1)
        unsqueeze_42 = None
        group_norm_23 = torch.nn.functional.group_norm(
            x_73,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_bias_ = (None)
        x_74 = torch.conv2d(
            group_norm_23,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_23 = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_75 = torch._C._nn.gelu(x_74, approximate="none")
        x_74 = None
        x_76 = torch.nn.functional.dropout(x_75, 0.0, False, False)
        x_75 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        mul_21 = unsqueeze_43 * x_78
        unsqueeze_43 = x_78 = None
        x_79 = x_73 + mul_21
        x_73 = mul_21 = None
        unsqueeze_44 = l_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_45 = unsqueeze_44.unsqueeze(-1)
        unsqueeze_44 = None
        group_norm_24 = torch.nn.functional.group_norm(
            x_79,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_11 = torch._C._nn.avg_pool2d(
            group_norm_24, 3, 1, 1, False, False, None
        )
        sub_11 = avg_pool2d_11 - group_norm_24
        avg_pool2d_11 = group_norm_24 = None
        mul_22 = unsqueeze_45 * sub_11
        unsqueeze_45 = sub_11 = None
        x_80 = x_79 + mul_22
        x_79 = mul_22 = None
        unsqueeze_46 = l_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_47 = unsqueeze_46.unsqueeze(-1)
        unsqueeze_46 = None
        group_norm_25 = torch.nn.functional.group_norm(
            x_80,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_bias_ = (None)
        x_81 = torch.conv2d(
            group_norm_25,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_25 = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_82 = torch._C._nn.gelu(x_81, approximate="none")
        x_81 = None
        x_83 = torch.nn.functional.dropout(x_82, 0.0, False, False)
        x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_83 = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_85 = torch.nn.functional.dropout(x_84, 0.0, False, False)
        x_84 = None
        mul_23 = unsqueeze_47 * x_85
        unsqueeze_47 = x_85 = None
        x_86 = x_80 + mul_23
        x_80 = mul_23 = None
        unsqueeze_48 = l_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_49 = unsqueeze_48.unsqueeze(-1)
        unsqueeze_48 = None
        group_norm_26 = torch.nn.functional.group_norm(
            x_86,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_12 = torch._C._nn.avg_pool2d(
            group_norm_26, 3, 1, 1, False, False, None
        )
        sub_12 = avg_pool2d_12 - group_norm_26
        avg_pool2d_12 = group_norm_26 = None
        mul_24 = unsqueeze_49 * sub_12
        unsqueeze_49 = sub_12 = None
        x_87 = x_86 + mul_24
        x_86 = mul_24 = None
        unsqueeze_50 = l_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_51 = unsqueeze_50.unsqueeze(-1)
        unsqueeze_50 = None
        group_norm_27 = torch.nn.functional.group_norm(
            x_87,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_bias_ = (None)
        x_88 = torch.conv2d(
            group_norm_27,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_27 = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_89 = torch._C._nn.gelu(x_88, approximate="none")
        x_88 = None
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_92 = torch.nn.functional.dropout(x_91, 0.0, False, False)
        x_91 = None
        mul_25 = unsqueeze_51 * x_92
        unsqueeze_51 = x_92 = None
        x_93 = x_87 + mul_25
        x_87 = mul_25 = None
        unsqueeze_52 = l_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_53 = unsqueeze_52.unsqueeze(-1)
        unsqueeze_52 = None
        group_norm_28 = torch.nn.functional.group_norm(
            x_93,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_13 = torch._C._nn.avg_pool2d(
            group_norm_28, 3, 1, 1, False, False, None
        )
        sub_13 = avg_pool2d_13 - group_norm_28
        avg_pool2d_13 = group_norm_28 = None
        mul_26 = unsqueeze_53 * sub_13
        unsqueeze_53 = sub_13 = None
        x_94 = x_93 + mul_26
        x_93 = mul_26 = None
        unsqueeze_54 = l_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_55 = unsqueeze_54.unsqueeze(-1)
        unsqueeze_54 = None
        group_norm_29 = torch.nn.functional.group_norm(
            x_94,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_bias_ = (None)
        x_95 = torch.conv2d(
            group_norm_29,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_29 = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_96 = torch._C._nn.gelu(x_95, approximate="none")
        x_95 = None
        x_97 = torch.nn.functional.dropout(x_96, 0.0, False, False)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_99 = torch.nn.functional.dropout(x_98, 0.0, False, False)
        x_98 = None
        mul_27 = unsqueeze_55 * x_99
        unsqueeze_55 = x_99 = None
        x_100 = x_94 + mul_27
        x_94 = mul_27 = None
        unsqueeze_56 = l_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_57 = unsqueeze_56.unsqueeze(-1)
        unsqueeze_56 = None
        group_norm_30 = torch.nn.functional.group_norm(
            x_100,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_14 = torch._C._nn.avg_pool2d(
            group_norm_30, 3, 1, 1, False, False, None
        )
        sub_14 = avg_pool2d_14 - group_norm_30
        avg_pool2d_14 = group_norm_30 = None
        mul_28 = unsqueeze_57 * sub_14
        unsqueeze_57 = sub_14 = None
        x_101 = x_100 + mul_28
        x_100 = mul_28 = None
        unsqueeze_58 = l_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_59 = unsqueeze_58.unsqueeze(-1)
        unsqueeze_58 = None
        group_norm_31 = torch.nn.functional.group_norm(
            x_101,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_bias_ = (None)
        x_102 = torch.conv2d(
            group_norm_31,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_31 = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_103 = torch._C._nn.gelu(x_102, approximate="none")
        x_102 = None
        x_104 = torch.nn.functional.dropout(x_103, 0.0, False, False)
        x_103 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        mul_29 = unsqueeze_59 * x_106
        unsqueeze_59 = x_106 = None
        x_107 = x_101 + mul_29
        x_101 = mul_29 = None
        unsqueeze_60 = l_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_61 = unsqueeze_60.unsqueeze(-1)
        unsqueeze_60 = None
        group_norm_32 = torch.nn.functional.group_norm(
            x_107,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_15 = torch._C._nn.avg_pool2d(
            group_norm_32, 3, 1, 1, False, False, None
        )
        sub_15 = avg_pool2d_15 - group_norm_32
        avg_pool2d_15 = group_norm_32 = None
        mul_30 = unsqueeze_61 * sub_15
        unsqueeze_61 = sub_15 = None
        x_108 = x_107 + mul_30
        x_107 = mul_30 = None
        unsqueeze_62 = l_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_63 = unsqueeze_62.unsqueeze(-1)
        unsqueeze_62 = None
        group_norm_33 = torch.nn.functional.group_norm(
            x_108,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_bias_ = (None)
        x_109 = torch.conv2d(
            group_norm_33,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_33 = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_110 = torch._C._nn.gelu(x_109, approximate="none")
        x_109 = None
        x_111 = torch.nn.functional.dropout(x_110, 0.0, False, False)
        x_110 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_113 = torch.nn.functional.dropout(x_112, 0.0, False, False)
        x_112 = None
        mul_31 = unsqueeze_63 * x_113
        unsqueeze_63 = x_113 = None
        x_114 = x_108 + mul_31
        x_108 = mul_31 = None
        unsqueeze_64 = l_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_65 = unsqueeze_64.unsqueeze(-1)
        unsqueeze_64 = None
        group_norm_34 = torch.nn.functional.group_norm(
            x_114,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_16 = torch._C._nn.avg_pool2d(
            group_norm_34, 3, 1, 1, False, False, None
        )
        sub_16 = avg_pool2d_16 - group_norm_34
        avg_pool2d_16 = group_norm_34 = None
        mul_32 = unsqueeze_65 * sub_16
        unsqueeze_65 = sub_16 = None
        x_115 = x_114 + mul_32
        x_114 = mul_32 = None
        unsqueeze_66 = l_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_67 = unsqueeze_66.unsqueeze(-1)
        unsqueeze_66 = None
        group_norm_35 = torch.nn.functional.group_norm(
            x_115,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_bias_ = (None)
        x_116 = torch.conv2d(
            group_norm_35,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_35 = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_117 = torch._C._nn.gelu(x_116, approximate="none")
        x_116 = None
        x_118 = torch.nn.functional.dropout(x_117, 0.0, False, False)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        mul_33 = unsqueeze_67 * x_120
        unsqueeze_67 = x_120 = None
        x_121 = x_115 + mul_33
        x_115 = mul_33 = None
        unsqueeze_68 = l_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_69 = unsqueeze_68.unsqueeze(-1)
        unsqueeze_68 = None
        group_norm_36 = torch.nn.functional.group_norm(
            x_121,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_17 = torch._C._nn.avg_pool2d(
            group_norm_36, 3, 1, 1, False, False, None
        )
        sub_17 = avg_pool2d_17 - group_norm_36
        avg_pool2d_17 = group_norm_36 = None
        mul_34 = unsqueeze_69 * sub_17
        unsqueeze_69 = sub_17 = None
        x_122 = x_121 + mul_34
        x_121 = mul_34 = None
        unsqueeze_70 = l_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_71 = unsqueeze_70.unsqueeze(-1)
        unsqueeze_70 = None
        group_norm_37 = torch.nn.functional.group_norm(
            x_122,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_bias_ = (None)
        x_123 = torch.conv2d(
            group_norm_37,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_37 = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_124 = torch._C._nn.gelu(x_123, approximate="none")
        x_123 = None
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        mul_35 = unsqueeze_71 * x_127
        unsqueeze_71 = x_127 = None
        x_128 = x_122 + mul_35
        x_122 = mul_35 = None
        unsqueeze_72 = l_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_73 = unsqueeze_72.unsqueeze(-1)
        unsqueeze_72 = None
        group_norm_38 = torch.nn.functional.group_norm(
            x_128,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_18 = torch._C._nn.avg_pool2d(
            group_norm_38, 3, 1, 1, False, False, None
        )
        sub_18 = avg_pool2d_18 - group_norm_38
        avg_pool2d_18 = group_norm_38 = None
        mul_36 = unsqueeze_73 * sub_18
        unsqueeze_73 = sub_18 = None
        x_129 = x_128 + mul_36
        x_128 = mul_36 = None
        unsqueeze_74 = l_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_75 = unsqueeze_74.unsqueeze(-1)
        unsqueeze_74 = None
        group_norm_39 = torch.nn.functional.group_norm(
            x_129,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_bias_ = (None)
        x_130 = torch.conv2d(
            group_norm_39,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_39 = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_131 = torch._C._nn.gelu(x_130, approximate="none")
        x_130 = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_132 = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_134 = torch.nn.functional.dropout(x_133, 0.0, False, False)
        x_133 = None
        mul_37 = unsqueeze_75 * x_134
        unsqueeze_75 = x_134 = None
        x_135 = x_129 + mul_37
        x_129 = mul_37 = None
        unsqueeze_76 = l_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_77 = unsqueeze_76.unsqueeze(-1)
        unsqueeze_76 = None
        group_norm_40 = torch.nn.functional.group_norm(
            x_135,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_19 = torch._C._nn.avg_pool2d(
            group_norm_40, 3, 1, 1, False, False, None
        )
        sub_19 = avg_pool2d_19 - group_norm_40
        avg_pool2d_19 = group_norm_40 = None
        mul_38 = unsqueeze_77 * sub_19
        unsqueeze_77 = sub_19 = None
        x_136 = x_135 + mul_38
        x_135 = mul_38 = None
        unsqueeze_78 = l_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_79 = unsqueeze_78.unsqueeze(-1)
        unsqueeze_78 = None
        group_norm_41 = torch.nn.functional.group_norm(
            x_136,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_bias_ = (None)
        x_137 = torch.conv2d(
            group_norm_41,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_41 = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_138 = torch._C._nn.gelu(x_137, approximate="none")
        x_137 = None
        x_139 = torch.nn.functional.dropout(x_138, 0.0, False, False)
        x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_141 = torch.nn.functional.dropout(x_140, 0.0, False, False)
        x_140 = None
        mul_39 = unsqueeze_79 * x_141
        unsqueeze_79 = x_141 = None
        x_142 = x_136 + mul_39
        x_136 = mul_39 = None
        x_out_2 = torch.nn.functional.group_norm(
            x_142,
            1,
            l_self_modules_backbone_modules_norm4_parameters_weight_,
            l_self_modules_backbone_modules_norm4_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_norm4_parameters_weight_ = (
            l_self_modules_backbone_modules_norm4_parameters_bias_
        ) = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_bias_ = (None)
        unsqueeze_80 = l_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_81 = unsqueeze_80.unsqueeze(-1)
        unsqueeze_80 = None
        group_norm_43 = torch.nn.functional.group_norm(
            x_143,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_20 = torch._C._nn.avg_pool2d(
            group_norm_43, 3, 1, 1, False, False, None
        )
        sub_20 = avg_pool2d_20 - group_norm_43
        avg_pool2d_20 = group_norm_43 = None
        mul_40 = unsqueeze_81 * sub_20
        unsqueeze_81 = sub_20 = None
        x_144 = x_143 + mul_40
        x_143 = mul_40 = None
        unsqueeze_82 = l_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_83 = unsqueeze_82.unsqueeze(-1)
        unsqueeze_82 = None
        group_norm_44 = torch.nn.functional.group_norm(
            x_144,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_bias_ = (None)
        x_145 = torch.conv2d(
            group_norm_44,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_44 = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_146 = torch._C._nn.gelu(x_145, approximate="none")
        x_145 = None
        x_147 = torch.nn.functional.dropout(x_146, 0.0, False, False)
        x_146 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_149 = torch.nn.functional.dropout(x_148, 0.0, False, False)
        x_148 = None
        mul_41 = unsqueeze_83 * x_149
        unsqueeze_83 = x_149 = None
        x_150 = x_144 + mul_41
        x_144 = mul_41 = None
        unsqueeze_84 = l_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_85 = unsqueeze_84.unsqueeze(-1)
        unsqueeze_84 = None
        group_norm_45 = torch.nn.functional.group_norm(
            x_150,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_21 = torch._C._nn.avg_pool2d(
            group_norm_45, 3, 1, 1, False, False, None
        )
        sub_21 = avg_pool2d_21 - group_norm_45
        avg_pool2d_21 = group_norm_45 = None
        mul_42 = unsqueeze_85 * sub_21
        unsqueeze_85 = sub_21 = None
        x_151 = x_150 + mul_42
        x_150 = mul_42 = None
        unsqueeze_86 = l_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_87 = unsqueeze_86.unsqueeze(-1)
        unsqueeze_86 = None
        group_norm_46 = torch.nn.functional.group_norm(
            x_151,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_bias_ = (None)
        x_152 = torch.conv2d(
            group_norm_46,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_46 = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_153 = torch._C._nn.gelu(x_152, approximate="none")
        x_152 = None
        x_154 = torch.nn.functional.dropout(x_153, 0.0, False, False)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        mul_43 = unsqueeze_87 * x_156
        unsqueeze_87 = x_156 = None
        x_157 = x_151 + mul_43
        x_151 = mul_43 = None
        unsqueeze_88 = l_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_89 = unsqueeze_88.unsqueeze(-1)
        unsqueeze_88 = None
        group_norm_47 = torch.nn.functional.group_norm(
            x_157,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_22 = torch._C._nn.avg_pool2d(
            group_norm_47, 3, 1, 1, False, False, None
        )
        sub_22 = avg_pool2d_22 - group_norm_47
        avg_pool2d_22 = group_norm_47 = None
        mul_44 = unsqueeze_89 * sub_22
        unsqueeze_89 = sub_22 = None
        x_158 = x_157 + mul_44
        x_157 = mul_44 = None
        unsqueeze_90 = l_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_91 = unsqueeze_90.unsqueeze(-1)
        unsqueeze_90 = None
        group_norm_48 = torch.nn.functional.group_norm(
            x_158,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_bias_ = (None)
        x_159 = torch.conv2d(
            group_norm_48,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_48 = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_160 = torch._C._nn.gelu(x_159, approximate="none")
        x_159 = None
        x_161 = torch.nn.functional.dropout(x_160, 0.0, False, False)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        mul_45 = unsqueeze_91 * x_163
        unsqueeze_91 = x_163 = None
        x_164 = x_158 + mul_45
        x_158 = mul_45 = None
        unsqueeze_92 = l_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_93 = unsqueeze_92.unsqueeze(-1)
        unsqueeze_92 = None
        group_norm_49 = torch.nn.functional.group_norm(
            x_164,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_23 = torch._C._nn.avg_pool2d(
            group_norm_49, 3, 1, 1, False, False, None
        )
        sub_23 = avg_pool2d_23 - group_norm_49
        avg_pool2d_23 = group_norm_49 = None
        mul_46 = unsqueeze_93 * sub_23
        unsqueeze_93 = sub_23 = None
        x_165 = x_164 + mul_46
        x_164 = mul_46 = None
        unsqueeze_94 = l_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_95 = unsqueeze_94.unsqueeze(-1)
        unsqueeze_94 = None
        group_norm_50 = torch.nn.functional.group_norm(
            x_165,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_bias_ = (None)
        x_166 = torch.conv2d(
            group_norm_50,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_50 = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_167 = torch._C._nn.gelu(x_166, approximate="none")
        x_166 = None
        x_168 = torch.nn.functional.dropout(x_167, 0.0, False, False)
        x_167 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_170 = torch.nn.functional.dropout(x_169, 0.0, False, False)
        x_169 = None
        mul_47 = unsqueeze_95 * x_170
        unsqueeze_95 = x_170 = None
        x_171 = x_165 + mul_47
        x_165 = mul_47 = None
        x_out_3 = torch.nn.functional.group_norm(
            x_171,
            1,
            l_self_modules_backbone_modules_norm6_parameters_weight_,
            l_self_modules_backbone_modules_norm6_parameters_bias_,
            1e-05,
        )
        x_171 = (
            l_self_modules_backbone_modules_norm6_parameters_weight_
        ) = l_self_modules_backbone_modules_norm6_parameters_bias_ = None
        x_172 = torch.conv2d(
            x_out,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out = l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_ = (None)
        x_173 = torch.conv2d(
            x_out_1,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_1 = l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_ = (None)
        x_174 = torch.conv2d(
            x_out_2,
            l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_2 = l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_ = (None)
        x_175 = torch.conv2d(
            x_out_3,
            l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_3 = l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_3_modules_conv_parameters_bias_ = (None)
        interpolate = torch.nn.functional.interpolate(
            x_175, (32, 32), None, "nearest", None
        )
        add_48 = x_174 + interpolate
        x_174 = interpolate = None
        interpolate_1 = torch.nn.functional.interpolate(
            add_48, (64, 64), None, "nearest", None
        )
        add_49 = x_173 + interpolate_1
        x_173 = interpolate_1 = None
        interpolate_2 = torch.nn.functional.interpolate(
            add_49, (128, 128), None, "nearest", None
        )
        add_50 = x_172 + interpolate_2
        x_172 = interpolate_2 = None
        x_176 = torch.conv2d(
            add_50,
            l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_50 = l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_ = (None)
        x_177 = torch.conv2d(
            add_49,
            l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_49 = l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_ = (None)
        x_178 = torch.conv2d(
            add_48,
            l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_48 = l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_ = (None)
        x_179 = torch.conv2d(
            x_175,
            l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_ = (None)
        x_180 = torch.conv2d(
            x_176,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_176 = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        x_182 = torch.nn.functional.relu(x_181, inplace=True)
        x_181 = None
        x_183 = torch.conv2d(
            x_177,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        x_185 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        input_1 = torch.nn.functional.interpolate(
            x_185, [128, 128], None, "bilinear", False
        )
        x_185 = None
        interpolate_4 = torch.nn.functional.interpolate(
            input_1, (128, 128), None, "bilinear", False
        )
        input_1 = None
        output = x_182 + interpolate_4
        x_182 = interpolate_4 = None
        x_186 = torch.conv2d(
            x_178,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_178 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_ = (None)
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        x_188 = torch.nn.functional.relu(x_187, inplace=True)
        x_187 = None
        input_2 = torch.nn.functional.interpolate(
            x_188, [64, 64], None, "bilinear", False
        )
        x_188 = None
        x_189 = torch.conv2d(
            input_2,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_ = (None)
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_189 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_ = (None)
        x_191 = torch.nn.functional.relu(x_190, inplace=True)
        x_190 = None
        input_3 = torch.nn.functional.interpolate(
            x_191, [128, 128], None, "bilinear", False
        )
        x_191 = None
        interpolate_7 = torch.nn.functional.interpolate(
            input_3, (128, 128), None, "bilinear", False
        )
        input_3 = None
        output_1 = output + interpolate_7
        output = interpolate_7 = None
        x_192 = torch.conv2d(
            x_179,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_ = (None)
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_192 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_ = (None)
        x_194 = torch.nn.functional.relu(x_193, inplace=True)
        x_193 = None
        input_4 = torch.nn.functional.interpolate(
            x_194, [32, 32], None, "bilinear", False
        )
        x_194 = None
        x_195 = torch.conv2d(
            input_4,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_ = (None)
        x_196 = torch.nn.functional.batch_norm(
            x_195,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_195 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_ = (None)
        x_197 = torch.nn.functional.relu(x_196, inplace=True)
        x_196 = None
        input_5 = torch.nn.functional.interpolate(
            x_197, [64, 64], None, "bilinear", False
        )
        x_197 = None
        x_198 = torch.conv2d(
            input_5,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_ = (None)
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_198 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_ = (None)
        x_200 = torch.nn.functional.relu(x_199, inplace=True)
        x_199 = None
        input_6 = torch.nn.functional.interpolate(
            x_200, [128, 128], None, "bilinear", False
        )
        x_200 = None
        interpolate_11 = torch.nn.functional.interpolate(
            input_6, (128, 128), None, "bilinear", False
        )
        input_6 = None
        output_2 = output_1 + interpolate_11
        output_1 = interpolate_11 = None
        feat = torch.nn.functional.dropout2d(output_2, 0.1, False, False)
        output_2 = None
        output_3 = torch.conv2d(
            feat,
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_,
            l_self_modules_decode_head_modules_conv_seg_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        feat = (
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_
        ) = l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = None
        return (output_3,)
