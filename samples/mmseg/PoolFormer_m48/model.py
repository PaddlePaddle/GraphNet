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
        L_self_modules_backbone_modules_network_modules_0_modules_4_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_4_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_5_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_5_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_6_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_6_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_7_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_7_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_network_modules_2_modules_4_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_4_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_5_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_5_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_6_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_6_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_7_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_7_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_network_modules_4_modules_12_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_12_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_13_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_13_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_14_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_14_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_15_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_15_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_16_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_16_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_17_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_17_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_18_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_18_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_19_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_19_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_20_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_20_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_21_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_21_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_22_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_22_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_23_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_23_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_network_modules_6_modules_4_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_4_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_5_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_5_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_6_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_6_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_7_parameters_layer_scale_1_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_7_parameters_layer_scale_2_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_network_modules_0_modules_4_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_0_modules_4_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_4_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_0_modules_4_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_5_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_0_modules_5_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_5_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_0_modules_5_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_6_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_0_modules_6_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_6_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_0_modules_6_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_7_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_0_modules_7_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_7_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_0_modules_7_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_network_modules_2_modules_4_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_2_modules_4_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_4_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_2_modules_4_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_5_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_2_modules_5_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_5_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_2_modules_5_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_6_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_2_modules_6_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_6_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_2_modules_6_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_7_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_2_modules_7_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_7_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_2_modules_7_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_network_modules_4_modules_12_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_12_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_12_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_12_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_13_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_13_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_13_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_13_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_14_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_14_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_14_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_14_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_15_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_15_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_15_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_15_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_16_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_16_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_16_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_16_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_17_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_17_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_17_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_17_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_18_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_18_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_18_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_18_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_19_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_19_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_19_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_19_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_20_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_20_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_20_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_20_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_21_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_21_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_21_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_21_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_22_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_22_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_22_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_22_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_23_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_4_modules_23_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_23_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_4_modules_23_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_network_modules_6_modules_4_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_6_modules_4_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_4_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_6_modules_4_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_5_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_6_modules_5_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_5_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_6_modules_5_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_6_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_6_modules_6_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_6_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_6_modules_6_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_7_parameters_layer_scale_1_ = L_self_modules_backbone_modules_network_modules_6_modules_7_parameters_layer_scale_1_
        l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_7_parameters_layer_scale_2_ = L_self_modules_backbone_modules_network_modules_6_modules_7_parameters_layer_scale_2_
        l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm2_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc2_parameters_bias_
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
        unsqueeze_16 = l_self_modules_backbone_modules_network_modules_0_modules_4_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_4_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_17 = unsqueeze_16.unsqueeze(-1)
        unsqueeze_16 = None
        group_norm_8 = torch.nn.functional.group_norm(
            x_28,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_4 = torch._C._nn.avg_pool2d(
            group_norm_8, 3, 1, 1, False, False, None
        )
        sub_4 = avg_pool2d_4 - group_norm_8
        avg_pool2d_4 = group_norm_8 = None
        mul_8 = unsqueeze_17 * sub_4
        unsqueeze_17 = sub_4 = None
        x_29 = x_28 + mul_8
        x_28 = mul_8 = None
        unsqueeze_18 = l_self_modules_backbone_modules_network_modules_0_modules_4_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_4_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_19 = unsqueeze_18.unsqueeze(-1)
        unsqueeze_18 = None
        group_norm_9 = torch.nn.functional.group_norm(
            x_29,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_4_modules_norm2_parameters_bias_ = (None)
        x_30 = torch.conv2d(
            group_norm_9,
            l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_9 = l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_31 = torch._C._nn.gelu(x_30, approximate="none")
        x_30 = None
        x_32 = torch.nn.functional.dropout(x_31, 0.0, False, False)
        x_31 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_34 = torch.nn.functional.dropout(x_33, 0.0, False, False)
        x_33 = None
        mul_9 = unsqueeze_19 * x_34
        unsqueeze_19 = x_34 = None
        x_35 = x_29 + mul_9
        x_29 = mul_9 = None
        unsqueeze_20 = l_self_modules_backbone_modules_network_modules_0_modules_5_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_5_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_21 = unsqueeze_20.unsqueeze(-1)
        unsqueeze_20 = None
        group_norm_10 = torch.nn.functional.group_norm(
            x_35,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_5 = torch._C._nn.avg_pool2d(
            group_norm_10, 3, 1, 1, False, False, None
        )
        sub_5 = avg_pool2d_5 - group_norm_10
        avg_pool2d_5 = group_norm_10 = None
        mul_10 = unsqueeze_21 * sub_5
        unsqueeze_21 = sub_5 = None
        x_36 = x_35 + mul_10
        x_35 = mul_10 = None
        unsqueeze_22 = l_self_modules_backbone_modules_network_modules_0_modules_5_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_5_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_23 = unsqueeze_22.unsqueeze(-1)
        unsqueeze_22 = None
        group_norm_11 = torch.nn.functional.group_norm(
            x_36,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_5_modules_norm2_parameters_bias_ = (None)
        x_37 = torch.conv2d(
            group_norm_11,
            l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_11 = l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_38 = torch._C._nn.gelu(x_37, approximate="none")
        x_37 = None
        x_39 = torch.nn.functional.dropout(x_38, 0.0, False, False)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_41 = torch.nn.functional.dropout(x_40, 0.0, False, False)
        x_40 = None
        mul_11 = unsqueeze_23 * x_41
        unsqueeze_23 = x_41 = None
        x_42 = x_36 + mul_11
        x_36 = mul_11 = None
        unsqueeze_24 = l_self_modules_backbone_modules_network_modules_0_modules_6_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_6_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_25 = unsqueeze_24.unsqueeze(-1)
        unsqueeze_24 = None
        group_norm_12 = torch.nn.functional.group_norm(
            x_42,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_6 = torch._C._nn.avg_pool2d(
            group_norm_12, 3, 1, 1, False, False, None
        )
        sub_6 = avg_pool2d_6 - group_norm_12
        avg_pool2d_6 = group_norm_12 = None
        mul_12 = unsqueeze_25 * sub_6
        unsqueeze_25 = sub_6 = None
        x_43 = x_42 + mul_12
        x_42 = mul_12 = None
        unsqueeze_26 = l_self_modules_backbone_modules_network_modules_0_modules_6_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_6_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_27 = unsqueeze_26.unsqueeze(-1)
        unsqueeze_26 = None
        group_norm_13 = torch.nn.functional.group_norm(
            x_43,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_6_modules_norm2_parameters_bias_ = (None)
        x_44 = torch.conv2d(
            group_norm_13,
            l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_13 = l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_45 = torch._C._nn.gelu(x_44, approximate="none")
        x_44 = None
        x_46 = torch.nn.functional.dropout(x_45, 0.0, False, False)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_48 = torch.nn.functional.dropout(x_47, 0.0, False, False)
        x_47 = None
        mul_13 = unsqueeze_27 * x_48
        unsqueeze_27 = x_48 = None
        x_49 = x_43 + mul_13
        x_43 = mul_13 = None
        unsqueeze_28 = l_self_modules_backbone_modules_network_modules_0_modules_7_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_7_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_29 = unsqueeze_28.unsqueeze(-1)
        unsqueeze_28 = None
        group_norm_14 = torch.nn.functional.group_norm(
            x_49,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_7 = torch._C._nn.avg_pool2d(
            group_norm_14, 3, 1, 1, False, False, None
        )
        sub_7 = avg_pool2d_7 - group_norm_14
        avg_pool2d_7 = group_norm_14 = None
        mul_14 = unsqueeze_29 * sub_7
        unsqueeze_29 = sub_7 = None
        x_50 = x_49 + mul_14
        x_49 = mul_14 = None
        unsqueeze_30 = l_self_modules_backbone_modules_network_modules_0_modules_7_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_0_modules_7_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_31 = unsqueeze_30.unsqueeze(-1)
        unsqueeze_30 = None
        group_norm_15 = torch.nn.functional.group_norm(
            x_50,
            1,
            l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_7_modules_norm2_parameters_bias_ = (None)
        x_51 = torch.conv2d(
            group_norm_15,
            l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_15 = l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_52 = torch._C._nn.gelu(x_51, approximate="none")
        x_51 = None
        x_53 = torch.nn.functional.dropout(x_52, 0.0, False, False)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_0_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_55 = torch.nn.functional.dropout(x_54, 0.0, False, False)
        x_54 = None
        mul_15 = unsqueeze_31 * x_55
        unsqueeze_31 = x_55 = None
        x_56 = x_50 + mul_15
        x_50 = mul_15 = None
        x_out = torch.nn.functional.group_norm(
            x_56,
            1,
            l_self_modules_backbone_modules_norm0_parameters_weight_,
            l_self_modules_backbone_modules_norm0_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_norm0_parameters_weight_ = (
            l_self_modules_backbone_modules_norm0_parameters_bias_
        ) = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_56 = l_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_network_modules_1_modules_proj_parameters_bias_ = (None)
        unsqueeze_32 = l_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_33 = unsqueeze_32.unsqueeze(-1)
        unsqueeze_32 = None
        group_norm_17 = torch.nn.functional.group_norm(
            x_57,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_8 = torch._C._nn.avg_pool2d(
            group_norm_17, 3, 1, 1, False, False, None
        )
        sub_8 = avg_pool2d_8 - group_norm_17
        avg_pool2d_8 = group_norm_17 = None
        mul_16 = unsqueeze_33 * sub_8
        unsqueeze_33 = sub_8 = None
        x_58 = x_57 + mul_16
        x_57 = mul_16 = None
        unsqueeze_34 = l_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_35 = unsqueeze_34.unsqueeze(-1)
        unsqueeze_34 = None
        group_norm_18 = torch.nn.functional.group_norm(
            x_58,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_norm2_parameters_bias_ = (None)
        x_59 = torch.conv2d(
            group_norm_18,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_18 = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_60 = torch._C._nn.gelu(x_59, approximate="none")
        x_59 = None
        x_61 = torch.nn.functional.dropout(x_60, 0.0, False, False)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_63 = torch.nn.functional.dropout(x_62, 0.0, False, False)
        x_62 = None
        mul_17 = unsqueeze_35 * x_63
        unsqueeze_35 = x_63 = None
        x_64 = x_58 + mul_17
        x_58 = mul_17 = None
        unsqueeze_36 = l_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_37 = unsqueeze_36.unsqueeze(-1)
        unsqueeze_36 = None
        group_norm_19 = torch.nn.functional.group_norm(
            x_64,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_9 = torch._C._nn.avg_pool2d(
            group_norm_19, 3, 1, 1, False, False, None
        )
        sub_9 = avg_pool2d_9 - group_norm_19
        avg_pool2d_9 = group_norm_19 = None
        mul_18 = unsqueeze_37 * sub_9
        unsqueeze_37 = sub_9 = None
        x_65 = x_64 + mul_18
        x_64 = mul_18 = None
        unsqueeze_38 = l_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_39 = unsqueeze_38.unsqueeze(-1)
        unsqueeze_38 = None
        group_norm_20 = torch.nn.functional.group_norm(
            x_65,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_norm2_parameters_bias_ = (None)
        x_66 = torch.conv2d(
            group_norm_20,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_20 = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_67 = torch._C._nn.gelu(x_66, approximate="none")
        x_66 = None
        x_68 = torch.nn.functional.dropout(x_67, 0.0, False, False)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        mul_19 = unsqueeze_39 * x_70
        unsqueeze_39 = x_70 = None
        x_71 = x_65 + mul_19
        x_65 = mul_19 = None
        unsqueeze_40 = l_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_41 = unsqueeze_40.unsqueeze(-1)
        unsqueeze_40 = None
        group_norm_21 = torch.nn.functional.group_norm(
            x_71,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_10 = torch._C._nn.avg_pool2d(
            group_norm_21, 3, 1, 1, False, False, None
        )
        sub_10 = avg_pool2d_10 - group_norm_21
        avg_pool2d_10 = group_norm_21 = None
        mul_20 = unsqueeze_41 * sub_10
        unsqueeze_41 = sub_10 = None
        x_72 = x_71 + mul_20
        x_71 = mul_20 = None
        unsqueeze_42 = l_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_43 = unsqueeze_42.unsqueeze(-1)
        unsqueeze_42 = None
        group_norm_22 = torch.nn.functional.group_norm(
            x_72,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_norm2_parameters_bias_ = (None)
        x_73 = torch.conv2d(
            group_norm_22,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_22 = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_74 = torch._C._nn.gelu(x_73, approximate="none")
        x_73 = None
        x_75 = torch.nn.functional.dropout(x_74, 0.0, False, False)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_77 = torch.nn.functional.dropout(x_76, 0.0, False, False)
        x_76 = None
        mul_21 = unsqueeze_43 * x_77
        unsqueeze_43 = x_77 = None
        x_78 = x_72 + mul_21
        x_72 = mul_21 = None
        unsqueeze_44 = l_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_45 = unsqueeze_44.unsqueeze(-1)
        unsqueeze_44 = None
        group_norm_23 = torch.nn.functional.group_norm(
            x_78,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_11 = torch._C._nn.avg_pool2d(
            group_norm_23, 3, 1, 1, False, False, None
        )
        sub_11 = avg_pool2d_11 - group_norm_23
        avg_pool2d_11 = group_norm_23 = None
        mul_22 = unsqueeze_45 * sub_11
        unsqueeze_45 = sub_11 = None
        x_79 = x_78 + mul_22
        x_78 = mul_22 = None
        unsqueeze_46 = l_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_3_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_47 = unsqueeze_46.unsqueeze(-1)
        unsqueeze_46 = None
        group_norm_24 = torch.nn.functional.group_norm(
            x_79,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_norm2_parameters_bias_ = (None)
        x_80 = torch.conv2d(
            group_norm_24,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_24 = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_81 = torch._C._nn.gelu(x_80, approximate="none")
        x_80 = None
        x_82 = torch.nn.functional.dropout(x_81, 0.0, False, False)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_84 = torch.nn.functional.dropout(x_83, 0.0, False, False)
        x_83 = None
        mul_23 = unsqueeze_47 * x_84
        unsqueeze_47 = x_84 = None
        x_85 = x_79 + mul_23
        x_79 = mul_23 = None
        unsqueeze_48 = l_self_modules_backbone_modules_network_modules_2_modules_4_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_4_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_49 = unsqueeze_48.unsqueeze(-1)
        unsqueeze_48 = None
        group_norm_25 = torch.nn.functional.group_norm(
            x_85,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_12 = torch._C._nn.avg_pool2d(
            group_norm_25, 3, 1, 1, False, False, None
        )
        sub_12 = avg_pool2d_12 - group_norm_25
        avg_pool2d_12 = group_norm_25 = None
        mul_24 = unsqueeze_49 * sub_12
        unsqueeze_49 = sub_12 = None
        x_86 = x_85 + mul_24
        x_85 = mul_24 = None
        unsqueeze_50 = l_self_modules_backbone_modules_network_modules_2_modules_4_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_4_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_51 = unsqueeze_50.unsqueeze(-1)
        unsqueeze_50 = None
        group_norm_26 = torch.nn.functional.group_norm(
            x_86,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_4_modules_norm2_parameters_bias_ = (None)
        x_87 = torch.conv2d(
            group_norm_26,
            l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_26 = l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_88 = torch._C._nn.gelu(x_87, approximate="none")
        x_87 = None
        x_89 = torch.nn.functional.dropout(x_88, 0.0, False, False)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        mul_25 = unsqueeze_51 * x_91
        unsqueeze_51 = x_91 = None
        x_92 = x_86 + mul_25
        x_86 = mul_25 = None
        unsqueeze_52 = l_self_modules_backbone_modules_network_modules_2_modules_5_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_5_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_53 = unsqueeze_52.unsqueeze(-1)
        unsqueeze_52 = None
        group_norm_27 = torch.nn.functional.group_norm(
            x_92,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_13 = torch._C._nn.avg_pool2d(
            group_norm_27, 3, 1, 1, False, False, None
        )
        sub_13 = avg_pool2d_13 - group_norm_27
        avg_pool2d_13 = group_norm_27 = None
        mul_26 = unsqueeze_53 * sub_13
        unsqueeze_53 = sub_13 = None
        x_93 = x_92 + mul_26
        x_92 = mul_26 = None
        unsqueeze_54 = l_self_modules_backbone_modules_network_modules_2_modules_5_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_5_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_55 = unsqueeze_54.unsqueeze(-1)
        unsqueeze_54 = None
        group_norm_28 = torch.nn.functional.group_norm(
            x_93,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_5_modules_norm2_parameters_bias_ = (None)
        x_94 = torch.conv2d(
            group_norm_28,
            l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_28 = l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_95 = torch._C._nn.gelu(x_94, approximate="none")
        x_94 = None
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_98 = torch.nn.functional.dropout(x_97, 0.0, False, False)
        x_97 = None
        mul_27 = unsqueeze_55 * x_98
        unsqueeze_55 = x_98 = None
        x_99 = x_93 + mul_27
        x_93 = mul_27 = None
        unsqueeze_56 = l_self_modules_backbone_modules_network_modules_2_modules_6_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_6_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_57 = unsqueeze_56.unsqueeze(-1)
        unsqueeze_56 = None
        group_norm_29 = torch.nn.functional.group_norm(
            x_99,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_14 = torch._C._nn.avg_pool2d(
            group_norm_29, 3, 1, 1, False, False, None
        )
        sub_14 = avg_pool2d_14 - group_norm_29
        avg_pool2d_14 = group_norm_29 = None
        mul_28 = unsqueeze_57 * sub_14
        unsqueeze_57 = sub_14 = None
        x_100 = x_99 + mul_28
        x_99 = mul_28 = None
        unsqueeze_58 = l_self_modules_backbone_modules_network_modules_2_modules_6_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_6_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_59 = unsqueeze_58.unsqueeze(-1)
        unsqueeze_58 = None
        group_norm_30 = torch.nn.functional.group_norm(
            x_100,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_6_modules_norm2_parameters_bias_ = (None)
        x_101 = torch.conv2d(
            group_norm_30,
            l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_30 = l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_102 = torch._C._nn.gelu(x_101, approximate="none")
        x_101 = None
        x_103 = torch.nn.functional.dropout(x_102, 0.0, False, False)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_105 = torch.nn.functional.dropout(x_104, 0.0, False, False)
        x_104 = None
        mul_29 = unsqueeze_59 * x_105
        unsqueeze_59 = x_105 = None
        x_106 = x_100 + mul_29
        x_100 = mul_29 = None
        unsqueeze_60 = l_self_modules_backbone_modules_network_modules_2_modules_7_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_7_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_61 = unsqueeze_60.unsqueeze(-1)
        unsqueeze_60 = None
        group_norm_31 = torch.nn.functional.group_norm(
            x_106,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_15 = torch._C._nn.avg_pool2d(
            group_norm_31, 3, 1, 1, False, False, None
        )
        sub_15 = avg_pool2d_15 - group_norm_31
        avg_pool2d_15 = group_norm_31 = None
        mul_30 = unsqueeze_61 * sub_15
        unsqueeze_61 = sub_15 = None
        x_107 = x_106 + mul_30
        x_106 = mul_30 = None
        unsqueeze_62 = l_self_modules_backbone_modules_network_modules_2_modules_7_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_2_modules_7_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_63 = unsqueeze_62.unsqueeze(-1)
        unsqueeze_62 = None
        group_norm_32 = torch.nn.functional.group_norm(
            x_107,
            1,
            l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_7_modules_norm2_parameters_bias_ = (None)
        x_108 = torch.conv2d(
            group_norm_32,
            l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_32 = l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_109 = torch._C._nn.gelu(x_108, approximate="none")
        x_108 = None
        x_110 = torch.nn.functional.dropout(x_109, 0.0, False, False)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_110 = l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_2_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_112 = torch.nn.functional.dropout(x_111, 0.0, False, False)
        x_111 = None
        mul_31 = unsqueeze_63 * x_112
        unsqueeze_63 = x_112 = None
        x_113 = x_107 + mul_31
        x_107 = mul_31 = None
        x_out_1 = torch.nn.functional.group_norm(
            x_113,
            1,
            l_self_modules_backbone_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_norm2_parameters_weight_ = (
            l_self_modules_backbone_modules_norm2_parameters_bias_
        ) = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_network_modules_3_modules_proj_parameters_bias_ = (None)
        unsqueeze_64 = l_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_65 = unsqueeze_64.unsqueeze(-1)
        unsqueeze_64 = None
        group_norm_34 = torch.nn.functional.group_norm(
            x_114,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_16 = torch._C._nn.avg_pool2d(
            group_norm_34, 3, 1, 1, False, False, None
        )
        sub_16 = avg_pool2d_16 - group_norm_34
        avg_pool2d_16 = group_norm_34 = None
        mul_32 = unsqueeze_65 * sub_16
        unsqueeze_65 = sub_16 = None
        x_115 = x_114 + mul_32
        x_114 = mul_32 = None
        unsqueeze_66 = l_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_67 = unsqueeze_66.unsqueeze(-1)
        unsqueeze_66 = None
        group_norm_35 = torch.nn.functional.group_norm(
            x_115,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_norm2_parameters_bias_ = (None)
        x_116 = torch.conv2d(
            group_norm_35,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_35 = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_117 = torch._C._nn.gelu(x_116, approximate="none")
        x_116 = None
        x_118 = torch.nn.functional.dropout(x_117, 0.0, False, False)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        mul_33 = unsqueeze_67 * x_120
        unsqueeze_67 = x_120 = None
        x_121 = x_115 + mul_33
        x_115 = mul_33 = None
        unsqueeze_68 = l_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_69 = unsqueeze_68.unsqueeze(-1)
        unsqueeze_68 = None
        group_norm_36 = torch.nn.functional.group_norm(
            x_121,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_17 = torch._C._nn.avg_pool2d(
            group_norm_36, 3, 1, 1, False, False, None
        )
        sub_17 = avg_pool2d_17 - group_norm_36
        avg_pool2d_17 = group_norm_36 = None
        mul_34 = unsqueeze_69 * sub_17
        unsqueeze_69 = sub_17 = None
        x_122 = x_121 + mul_34
        x_121 = mul_34 = None
        unsqueeze_70 = l_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_71 = unsqueeze_70.unsqueeze(-1)
        unsqueeze_70 = None
        group_norm_37 = torch.nn.functional.group_norm(
            x_122,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_norm2_parameters_bias_ = (None)
        x_123 = torch.conv2d(
            group_norm_37,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_37 = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_124 = torch._C._nn.gelu(x_123, approximate="none")
        x_123 = None
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        mul_35 = unsqueeze_71 * x_127
        unsqueeze_71 = x_127 = None
        x_128 = x_122 + mul_35
        x_122 = mul_35 = None
        unsqueeze_72 = l_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_73 = unsqueeze_72.unsqueeze(-1)
        unsqueeze_72 = None
        group_norm_38 = torch.nn.functional.group_norm(
            x_128,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_18 = torch._C._nn.avg_pool2d(
            group_norm_38, 3, 1, 1, False, False, None
        )
        sub_18 = avg_pool2d_18 - group_norm_38
        avg_pool2d_18 = group_norm_38 = None
        mul_36 = unsqueeze_73 * sub_18
        unsqueeze_73 = sub_18 = None
        x_129 = x_128 + mul_36
        x_128 = mul_36 = None
        unsqueeze_74 = l_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_75 = unsqueeze_74.unsqueeze(-1)
        unsqueeze_74 = None
        group_norm_39 = torch.nn.functional.group_norm(
            x_129,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_norm2_parameters_bias_ = (None)
        x_130 = torch.conv2d(
            group_norm_39,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_39 = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_131 = torch._C._nn.gelu(x_130, approximate="none")
        x_130 = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_132 = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_134 = torch.nn.functional.dropout(x_133, 0.0, False, False)
        x_133 = None
        mul_37 = unsqueeze_75 * x_134
        unsqueeze_75 = x_134 = None
        x_135 = x_129 + mul_37
        x_129 = mul_37 = None
        unsqueeze_76 = l_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_77 = unsqueeze_76.unsqueeze(-1)
        unsqueeze_76 = None
        group_norm_40 = torch.nn.functional.group_norm(
            x_135,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_19 = torch._C._nn.avg_pool2d(
            group_norm_40, 3, 1, 1, False, False, None
        )
        sub_19 = avg_pool2d_19 - group_norm_40
        avg_pool2d_19 = group_norm_40 = None
        mul_38 = unsqueeze_77 * sub_19
        unsqueeze_77 = sub_19 = None
        x_136 = x_135 + mul_38
        x_135 = mul_38 = None
        unsqueeze_78 = l_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_3_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_79 = unsqueeze_78.unsqueeze(-1)
        unsqueeze_78 = None
        group_norm_41 = torch.nn.functional.group_norm(
            x_136,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_norm2_parameters_bias_ = (None)
        x_137 = torch.conv2d(
            group_norm_41,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_41 = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_138 = torch._C._nn.gelu(x_137, approximate="none")
        x_137 = None
        x_139 = torch.nn.functional.dropout(x_138, 0.0, False, False)
        x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_141 = torch.nn.functional.dropout(x_140, 0.0, False, False)
        x_140 = None
        mul_39 = unsqueeze_79 * x_141
        unsqueeze_79 = x_141 = None
        x_142 = x_136 + mul_39
        x_136 = mul_39 = None
        unsqueeze_80 = l_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_81 = unsqueeze_80.unsqueeze(-1)
        unsqueeze_80 = None
        group_norm_42 = torch.nn.functional.group_norm(
            x_142,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_20 = torch._C._nn.avg_pool2d(
            group_norm_42, 3, 1, 1, False, False, None
        )
        sub_20 = avg_pool2d_20 - group_norm_42
        avg_pool2d_20 = group_norm_42 = None
        mul_40 = unsqueeze_81 * sub_20
        unsqueeze_81 = sub_20 = None
        x_143 = x_142 + mul_40
        x_142 = mul_40 = None
        unsqueeze_82 = l_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_4_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_83 = unsqueeze_82.unsqueeze(-1)
        unsqueeze_82 = None
        group_norm_43 = torch.nn.functional.group_norm(
            x_143,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_norm2_parameters_bias_ = (None)
        x_144 = torch.conv2d(
            group_norm_43,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_43 = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_145 = torch._C._nn.gelu(x_144, approximate="none")
        x_144 = None
        x_146 = torch.nn.functional.dropout(x_145, 0.0, False, False)
        x_145 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        mul_41 = unsqueeze_83 * x_148
        unsqueeze_83 = x_148 = None
        x_149 = x_143 + mul_41
        x_143 = mul_41 = None
        unsqueeze_84 = l_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_85 = unsqueeze_84.unsqueeze(-1)
        unsqueeze_84 = None
        group_norm_44 = torch.nn.functional.group_norm(
            x_149,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_21 = torch._C._nn.avg_pool2d(
            group_norm_44, 3, 1, 1, False, False, None
        )
        sub_21 = avg_pool2d_21 - group_norm_44
        avg_pool2d_21 = group_norm_44 = None
        mul_42 = unsqueeze_85 * sub_21
        unsqueeze_85 = sub_21 = None
        x_150 = x_149 + mul_42
        x_149 = mul_42 = None
        unsqueeze_86 = l_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_5_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_87 = unsqueeze_86.unsqueeze(-1)
        unsqueeze_86 = None
        group_norm_45 = torch.nn.functional.group_norm(
            x_150,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_norm2_parameters_bias_ = (None)
        x_151 = torch.conv2d(
            group_norm_45,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_45 = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_152 = torch._C._nn.gelu(x_151, approximate="none")
        x_151 = None
        x_153 = torch.nn.functional.dropout(x_152, 0.0, False, False)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_155 = torch.nn.functional.dropout(x_154, 0.0, False, False)
        x_154 = None
        mul_43 = unsqueeze_87 * x_155
        unsqueeze_87 = x_155 = None
        x_156 = x_150 + mul_43
        x_150 = mul_43 = None
        unsqueeze_88 = l_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_89 = unsqueeze_88.unsqueeze(-1)
        unsqueeze_88 = None
        group_norm_46 = torch.nn.functional.group_norm(
            x_156,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_22 = torch._C._nn.avg_pool2d(
            group_norm_46, 3, 1, 1, False, False, None
        )
        sub_22 = avg_pool2d_22 - group_norm_46
        avg_pool2d_22 = group_norm_46 = None
        mul_44 = unsqueeze_89 * sub_22
        unsqueeze_89 = sub_22 = None
        x_157 = x_156 + mul_44
        x_156 = mul_44 = None
        unsqueeze_90 = l_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_6_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_91 = unsqueeze_90.unsqueeze(-1)
        unsqueeze_90 = None
        group_norm_47 = torch.nn.functional.group_norm(
            x_157,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_norm2_parameters_bias_ = (None)
        x_158 = torch.conv2d(
            group_norm_47,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_47 = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_159 = torch._C._nn.gelu(x_158, approximate="none")
        x_158 = None
        x_160 = torch.nn.functional.dropout(x_159, 0.0, False, False)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_162 = torch.nn.functional.dropout(x_161, 0.0, False, False)
        x_161 = None
        mul_45 = unsqueeze_91 * x_162
        unsqueeze_91 = x_162 = None
        x_163 = x_157 + mul_45
        x_157 = mul_45 = None
        unsqueeze_92 = l_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_93 = unsqueeze_92.unsqueeze(-1)
        unsqueeze_92 = None
        group_norm_48 = torch.nn.functional.group_norm(
            x_163,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_23 = torch._C._nn.avg_pool2d(
            group_norm_48, 3, 1, 1, False, False, None
        )
        sub_23 = avg_pool2d_23 - group_norm_48
        avg_pool2d_23 = group_norm_48 = None
        mul_46 = unsqueeze_93 * sub_23
        unsqueeze_93 = sub_23 = None
        x_164 = x_163 + mul_46
        x_163 = mul_46 = None
        unsqueeze_94 = l_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_7_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_95 = unsqueeze_94.unsqueeze(-1)
        unsqueeze_94 = None
        group_norm_49 = torch.nn.functional.group_norm(
            x_164,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_norm2_parameters_bias_ = (None)
        x_165 = torch.conv2d(
            group_norm_49,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_49 = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_166 = torch._C._nn.gelu(x_165, approximate="none")
        x_165 = None
        x_167 = torch.nn.functional.dropout(x_166, 0.0, False, False)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_169 = torch.nn.functional.dropout(x_168, 0.0, False, False)
        x_168 = None
        mul_47 = unsqueeze_95 * x_169
        unsqueeze_95 = x_169 = None
        x_170 = x_164 + mul_47
        x_164 = mul_47 = None
        unsqueeze_96 = l_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_97 = unsqueeze_96.unsqueeze(-1)
        unsqueeze_96 = None
        group_norm_50 = torch.nn.functional.group_norm(
            x_170,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_24 = torch._C._nn.avg_pool2d(
            group_norm_50, 3, 1, 1, False, False, None
        )
        sub_24 = avg_pool2d_24 - group_norm_50
        avg_pool2d_24 = group_norm_50 = None
        mul_48 = unsqueeze_97 * sub_24
        unsqueeze_97 = sub_24 = None
        x_171 = x_170 + mul_48
        x_170 = mul_48 = None
        unsqueeze_98 = l_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_8_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_99 = unsqueeze_98.unsqueeze(-1)
        unsqueeze_98 = None
        group_norm_51 = torch.nn.functional.group_norm(
            x_171,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_norm2_parameters_bias_ = (None)
        x_172 = torch.conv2d(
            group_norm_51,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_51 = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_173 = torch._C._nn.gelu(x_172, approximate="none")
        x_172 = None
        x_174 = torch.nn.functional.dropout(x_173, 0.0, False, False)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_176 = torch.nn.functional.dropout(x_175, 0.0, False, False)
        x_175 = None
        mul_49 = unsqueeze_99 * x_176
        unsqueeze_99 = x_176 = None
        x_177 = x_171 + mul_49
        x_171 = mul_49 = None
        unsqueeze_100 = l_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_101 = unsqueeze_100.unsqueeze(-1)
        unsqueeze_100 = None
        group_norm_52 = torch.nn.functional.group_norm(
            x_177,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_25 = torch._C._nn.avg_pool2d(
            group_norm_52, 3, 1, 1, False, False, None
        )
        sub_25 = avg_pool2d_25 - group_norm_52
        avg_pool2d_25 = group_norm_52 = None
        mul_50 = unsqueeze_101 * sub_25
        unsqueeze_101 = sub_25 = None
        x_178 = x_177 + mul_50
        x_177 = mul_50 = None
        unsqueeze_102 = l_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_9_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_103 = unsqueeze_102.unsqueeze(-1)
        unsqueeze_102 = None
        group_norm_53 = torch.nn.functional.group_norm(
            x_178,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_norm2_parameters_bias_ = (None)
        x_179 = torch.conv2d(
            group_norm_53,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_53 = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_180 = torch._C._nn.gelu(x_179, approximate="none")
        x_179 = None
        x_181 = torch.nn.functional.dropout(x_180, 0.0, False, False)
        x_180 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_183 = torch.nn.functional.dropout(x_182, 0.0, False, False)
        x_182 = None
        mul_51 = unsqueeze_103 * x_183
        unsqueeze_103 = x_183 = None
        x_184 = x_178 + mul_51
        x_178 = mul_51 = None
        unsqueeze_104 = l_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_105 = unsqueeze_104.unsqueeze(-1)
        unsqueeze_104 = None
        group_norm_54 = torch.nn.functional.group_norm(
            x_184,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_26 = torch._C._nn.avg_pool2d(
            group_norm_54, 3, 1, 1, False, False, None
        )
        sub_26 = avg_pool2d_26 - group_norm_54
        avg_pool2d_26 = group_norm_54 = None
        mul_52 = unsqueeze_105 * sub_26
        unsqueeze_105 = sub_26 = None
        x_185 = x_184 + mul_52
        x_184 = mul_52 = None
        unsqueeze_106 = l_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_10_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_107 = unsqueeze_106.unsqueeze(-1)
        unsqueeze_106 = None
        group_norm_55 = torch.nn.functional.group_norm(
            x_185,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_norm2_parameters_bias_ = (None)
        x_186 = torch.conv2d(
            group_norm_55,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_55 = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_187 = torch._C._nn.gelu(x_186, approximate="none")
        x_186 = None
        x_188 = torch.nn.functional.dropout(x_187, 0.0, False, False)
        x_187 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        mul_53 = unsqueeze_107 * x_190
        unsqueeze_107 = x_190 = None
        x_191 = x_185 + mul_53
        x_185 = mul_53 = None
        unsqueeze_108 = l_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_109 = unsqueeze_108.unsqueeze(-1)
        unsqueeze_108 = None
        group_norm_56 = torch.nn.functional.group_norm(
            x_191,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_27 = torch._C._nn.avg_pool2d(
            group_norm_56, 3, 1, 1, False, False, None
        )
        sub_27 = avg_pool2d_27 - group_norm_56
        avg_pool2d_27 = group_norm_56 = None
        mul_54 = unsqueeze_109 * sub_27
        unsqueeze_109 = sub_27 = None
        x_192 = x_191 + mul_54
        x_191 = mul_54 = None
        unsqueeze_110 = l_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_11_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_111 = unsqueeze_110.unsqueeze(-1)
        unsqueeze_110 = None
        group_norm_57 = torch.nn.functional.group_norm(
            x_192,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_norm2_parameters_bias_ = (None)
        x_193 = torch.conv2d(
            group_norm_57,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_57 = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_194 = torch._C._nn.gelu(x_193, approximate="none")
        x_193 = None
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_197 = torch.nn.functional.dropout(x_196, 0.0, False, False)
        x_196 = None
        mul_55 = unsqueeze_111 * x_197
        unsqueeze_111 = x_197 = None
        x_198 = x_192 + mul_55
        x_192 = mul_55 = None
        unsqueeze_112 = l_self_modules_backbone_modules_network_modules_4_modules_12_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_12_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_113 = unsqueeze_112.unsqueeze(-1)
        unsqueeze_112 = None
        group_norm_58 = torch.nn.functional.group_norm(
            x_198,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_28 = torch._C._nn.avg_pool2d(
            group_norm_58, 3, 1, 1, False, False, None
        )
        sub_28 = avg_pool2d_28 - group_norm_58
        avg_pool2d_28 = group_norm_58 = None
        mul_56 = unsqueeze_113 * sub_28
        unsqueeze_113 = sub_28 = None
        x_199 = x_198 + mul_56
        x_198 = mul_56 = None
        unsqueeze_114 = l_self_modules_backbone_modules_network_modules_4_modules_12_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_12_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_115 = unsqueeze_114.unsqueeze(-1)
        unsqueeze_114 = None
        group_norm_59 = torch.nn.functional.group_norm(
            x_199,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_12_modules_norm2_parameters_bias_ = (None)
        x_200 = torch.conv2d(
            group_norm_59,
            l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_59 = l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_201 = torch._C._nn.gelu(x_200, approximate="none")
        x_200 = None
        x_202 = torch.nn.functional.dropout(x_201, 0.0, False, False)
        x_201 = None
        x_203 = torch.conv2d(
            x_202,
            l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_202 = l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_204 = torch.nn.functional.dropout(x_203, 0.0, False, False)
        x_203 = None
        mul_57 = unsqueeze_115 * x_204
        unsqueeze_115 = x_204 = None
        x_205 = x_199 + mul_57
        x_199 = mul_57 = None
        unsqueeze_116 = l_self_modules_backbone_modules_network_modules_4_modules_13_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_13_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_117 = unsqueeze_116.unsqueeze(-1)
        unsqueeze_116 = None
        group_norm_60 = torch.nn.functional.group_norm(
            x_205,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_29 = torch._C._nn.avg_pool2d(
            group_norm_60, 3, 1, 1, False, False, None
        )
        sub_29 = avg_pool2d_29 - group_norm_60
        avg_pool2d_29 = group_norm_60 = None
        mul_58 = unsqueeze_117 * sub_29
        unsqueeze_117 = sub_29 = None
        x_206 = x_205 + mul_58
        x_205 = mul_58 = None
        unsqueeze_118 = l_self_modules_backbone_modules_network_modules_4_modules_13_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_13_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_119 = unsqueeze_118.unsqueeze(-1)
        unsqueeze_118 = None
        group_norm_61 = torch.nn.functional.group_norm(
            x_206,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_13_modules_norm2_parameters_bias_ = (None)
        x_207 = torch.conv2d(
            group_norm_61,
            l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_61 = l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_208 = torch._C._nn.gelu(x_207, approximate="none")
        x_207 = None
        x_209 = torch.nn.functional.dropout(x_208, 0.0, False, False)
        x_208 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_211 = torch.nn.functional.dropout(x_210, 0.0, False, False)
        x_210 = None
        mul_59 = unsqueeze_119 * x_211
        unsqueeze_119 = x_211 = None
        x_212 = x_206 + mul_59
        x_206 = mul_59 = None
        unsqueeze_120 = l_self_modules_backbone_modules_network_modules_4_modules_14_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_14_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_121 = unsqueeze_120.unsqueeze(-1)
        unsqueeze_120 = None
        group_norm_62 = torch.nn.functional.group_norm(
            x_212,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_30 = torch._C._nn.avg_pool2d(
            group_norm_62, 3, 1, 1, False, False, None
        )
        sub_30 = avg_pool2d_30 - group_norm_62
        avg_pool2d_30 = group_norm_62 = None
        mul_60 = unsqueeze_121 * sub_30
        unsqueeze_121 = sub_30 = None
        x_213 = x_212 + mul_60
        x_212 = mul_60 = None
        unsqueeze_122 = l_self_modules_backbone_modules_network_modules_4_modules_14_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_14_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_123 = unsqueeze_122.unsqueeze(-1)
        unsqueeze_122 = None
        group_norm_63 = torch.nn.functional.group_norm(
            x_213,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_14_modules_norm2_parameters_bias_ = (None)
        x_214 = torch.conv2d(
            group_norm_63,
            l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_63 = l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_215 = torch._C._nn.gelu(x_214, approximate="none")
        x_214 = None
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_216 = l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_218 = torch.nn.functional.dropout(x_217, 0.0, False, False)
        x_217 = None
        mul_61 = unsqueeze_123 * x_218
        unsqueeze_123 = x_218 = None
        x_219 = x_213 + mul_61
        x_213 = mul_61 = None
        unsqueeze_124 = l_self_modules_backbone_modules_network_modules_4_modules_15_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_15_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_125 = unsqueeze_124.unsqueeze(-1)
        unsqueeze_124 = None
        group_norm_64 = torch.nn.functional.group_norm(
            x_219,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_31 = torch._C._nn.avg_pool2d(
            group_norm_64, 3, 1, 1, False, False, None
        )
        sub_31 = avg_pool2d_31 - group_norm_64
        avg_pool2d_31 = group_norm_64 = None
        mul_62 = unsqueeze_125 * sub_31
        unsqueeze_125 = sub_31 = None
        x_220 = x_219 + mul_62
        x_219 = mul_62 = None
        unsqueeze_126 = l_self_modules_backbone_modules_network_modules_4_modules_15_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_15_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_127 = unsqueeze_126.unsqueeze(-1)
        unsqueeze_126 = None
        group_norm_65 = torch.nn.functional.group_norm(
            x_220,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_15_modules_norm2_parameters_bias_ = (None)
        x_221 = torch.conv2d(
            group_norm_65,
            l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_65 = l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_222 = torch._C._nn.gelu(x_221, approximate="none")
        x_221 = None
        x_223 = torch.nn.functional.dropout(x_222, 0.0, False, False)
        x_222 = None
        x_224 = torch.conv2d(
            x_223,
            l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_223 = l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_225 = torch.nn.functional.dropout(x_224, 0.0, False, False)
        x_224 = None
        mul_63 = unsqueeze_127 * x_225
        unsqueeze_127 = x_225 = None
        x_226 = x_220 + mul_63
        x_220 = mul_63 = None
        unsqueeze_128 = l_self_modules_backbone_modules_network_modules_4_modules_16_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_16_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_129 = unsqueeze_128.unsqueeze(-1)
        unsqueeze_128 = None
        group_norm_66 = torch.nn.functional.group_norm(
            x_226,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_32 = torch._C._nn.avg_pool2d(
            group_norm_66, 3, 1, 1, False, False, None
        )
        sub_32 = avg_pool2d_32 - group_norm_66
        avg_pool2d_32 = group_norm_66 = None
        mul_64 = unsqueeze_129 * sub_32
        unsqueeze_129 = sub_32 = None
        x_227 = x_226 + mul_64
        x_226 = mul_64 = None
        unsqueeze_130 = l_self_modules_backbone_modules_network_modules_4_modules_16_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_16_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_131 = unsqueeze_130.unsqueeze(-1)
        unsqueeze_130 = None
        group_norm_67 = torch.nn.functional.group_norm(
            x_227,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_16_modules_norm2_parameters_bias_ = (None)
        x_228 = torch.conv2d(
            group_norm_67,
            l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_67 = l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_229 = torch._C._nn.gelu(x_228, approximate="none")
        x_228 = None
        x_230 = torch.nn.functional.dropout(x_229, 0.0, False, False)
        x_229 = None
        x_231 = torch.conv2d(
            x_230,
            l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_230 = l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_232 = torch.nn.functional.dropout(x_231, 0.0, False, False)
        x_231 = None
        mul_65 = unsqueeze_131 * x_232
        unsqueeze_131 = x_232 = None
        x_233 = x_227 + mul_65
        x_227 = mul_65 = None
        unsqueeze_132 = l_self_modules_backbone_modules_network_modules_4_modules_17_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_17_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_133 = unsqueeze_132.unsqueeze(-1)
        unsqueeze_132 = None
        group_norm_68 = torch.nn.functional.group_norm(
            x_233,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_33 = torch._C._nn.avg_pool2d(
            group_norm_68, 3, 1, 1, False, False, None
        )
        sub_33 = avg_pool2d_33 - group_norm_68
        avg_pool2d_33 = group_norm_68 = None
        mul_66 = unsqueeze_133 * sub_33
        unsqueeze_133 = sub_33 = None
        x_234 = x_233 + mul_66
        x_233 = mul_66 = None
        unsqueeze_134 = l_self_modules_backbone_modules_network_modules_4_modules_17_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_17_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_135 = unsqueeze_134.unsqueeze(-1)
        unsqueeze_134 = None
        group_norm_69 = torch.nn.functional.group_norm(
            x_234,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_17_modules_norm2_parameters_bias_ = (None)
        x_235 = torch.conv2d(
            group_norm_69,
            l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_69 = l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_236 = torch._C._nn.gelu(x_235, approximate="none")
        x_235 = None
        x_237 = torch.nn.functional.dropout(x_236, 0.0, False, False)
        x_236 = None
        x_238 = torch.conv2d(
            x_237,
            l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_237 = l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_239 = torch.nn.functional.dropout(x_238, 0.0, False, False)
        x_238 = None
        mul_67 = unsqueeze_135 * x_239
        unsqueeze_135 = x_239 = None
        x_240 = x_234 + mul_67
        x_234 = mul_67 = None
        unsqueeze_136 = l_self_modules_backbone_modules_network_modules_4_modules_18_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_18_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_137 = unsqueeze_136.unsqueeze(-1)
        unsqueeze_136 = None
        group_norm_70 = torch.nn.functional.group_norm(
            x_240,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_34 = torch._C._nn.avg_pool2d(
            group_norm_70, 3, 1, 1, False, False, None
        )
        sub_34 = avg_pool2d_34 - group_norm_70
        avg_pool2d_34 = group_norm_70 = None
        mul_68 = unsqueeze_137 * sub_34
        unsqueeze_137 = sub_34 = None
        x_241 = x_240 + mul_68
        x_240 = mul_68 = None
        unsqueeze_138 = l_self_modules_backbone_modules_network_modules_4_modules_18_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_18_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_139 = unsqueeze_138.unsqueeze(-1)
        unsqueeze_138 = None
        group_norm_71 = torch.nn.functional.group_norm(
            x_241,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_18_modules_norm2_parameters_bias_ = (None)
        x_242 = torch.conv2d(
            group_norm_71,
            l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_71 = l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_243 = torch._C._nn.gelu(x_242, approximate="none")
        x_242 = None
        x_244 = torch.nn.functional.dropout(x_243, 0.0, False, False)
        x_243 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_244 = l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_246 = torch.nn.functional.dropout(x_245, 0.0, False, False)
        x_245 = None
        mul_69 = unsqueeze_139 * x_246
        unsqueeze_139 = x_246 = None
        x_247 = x_241 + mul_69
        x_241 = mul_69 = None
        unsqueeze_140 = l_self_modules_backbone_modules_network_modules_4_modules_19_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_19_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_141 = unsqueeze_140.unsqueeze(-1)
        unsqueeze_140 = None
        group_norm_72 = torch.nn.functional.group_norm(
            x_247,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_35 = torch._C._nn.avg_pool2d(
            group_norm_72, 3, 1, 1, False, False, None
        )
        sub_35 = avg_pool2d_35 - group_norm_72
        avg_pool2d_35 = group_norm_72 = None
        mul_70 = unsqueeze_141 * sub_35
        unsqueeze_141 = sub_35 = None
        x_248 = x_247 + mul_70
        x_247 = mul_70 = None
        unsqueeze_142 = l_self_modules_backbone_modules_network_modules_4_modules_19_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_19_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_143 = unsqueeze_142.unsqueeze(-1)
        unsqueeze_142 = None
        group_norm_73 = torch.nn.functional.group_norm(
            x_248,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_19_modules_norm2_parameters_bias_ = (None)
        x_249 = torch.conv2d(
            group_norm_73,
            l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_73 = l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_250 = torch._C._nn.gelu(x_249, approximate="none")
        x_249 = None
        x_251 = torch.nn.functional.dropout(x_250, 0.0, False, False)
        x_250 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_253 = torch.nn.functional.dropout(x_252, 0.0, False, False)
        x_252 = None
        mul_71 = unsqueeze_143 * x_253
        unsqueeze_143 = x_253 = None
        x_254 = x_248 + mul_71
        x_248 = mul_71 = None
        unsqueeze_144 = l_self_modules_backbone_modules_network_modules_4_modules_20_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_20_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_145 = unsqueeze_144.unsqueeze(-1)
        unsqueeze_144 = None
        group_norm_74 = torch.nn.functional.group_norm(
            x_254,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_36 = torch._C._nn.avg_pool2d(
            group_norm_74, 3, 1, 1, False, False, None
        )
        sub_36 = avg_pool2d_36 - group_norm_74
        avg_pool2d_36 = group_norm_74 = None
        mul_72 = unsqueeze_145 * sub_36
        unsqueeze_145 = sub_36 = None
        x_255 = x_254 + mul_72
        x_254 = mul_72 = None
        unsqueeze_146 = l_self_modules_backbone_modules_network_modules_4_modules_20_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_20_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_147 = unsqueeze_146.unsqueeze(-1)
        unsqueeze_146 = None
        group_norm_75 = torch.nn.functional.group_norm(
            x_255,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_20_modules_norm2_parameters_bias_ = (None)
        x_256 = torch.conv2d(
            group_norm_75,
            l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_75 = l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_257 = torch._C._nn.gelu(x_256, approximate="none")
        x_256 = None
        x_258 = torch.nn.functional.dropout(x_257, 0.0, False, False)
        x_257 = None
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_258 = l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_260 = torch.nn.functional.dropout(x_259, 0.0, False, False)
        x_259 = None
        mul_73 = unsqueeze_147 * x_260
        unsqueeze_147 = x_260 = None
        x_261 = x_255 + mul_73
        x_255 = mul_73 = None
        unsqueeze_148 = l_self_modules_backbone_modules_network_modules_4_modules_21_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_21_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_149 = unsqueeze_148.unsqueeze(-1)
        unsqueeze_148 = None
        group_norm_76 = torch.nn.functional.group_norm(
            x_261,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_37 = torch._C._nn.avg_pool2d(
            group_norm_76, 3, 1, 1, False, False, None
        )
        sub_37 = avg_pool2d_37 - group_norm_76
        avg_pool2d_37 = group_norm_76 = None
        mul_74 = unsqueeze_149 * sub_37
        unsqueeze_149 = sub_37 = None
        x_262 = x_261 + mul_74
        x_261 = mul_74 = None
        unsqueeze_150 = l_self_modules_backbone_modules_network_modules_4_modules_21_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_21_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_151 = unsqueeze_150.unsqueeze(-1)
        unsqueeze_150 = None
        group_norm_77 = torch.nn.functional.group_norm(
            x_262,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_21_modules_norm2_parameters_bias_ = (None)
        x_263 = torch.conv2d(
            group_norm_77,
            l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_77 = l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_264 = torch._C._nn.gelu(x_263, approximate="none")
        x_263 = None
        x_265 = torch.nn.functional.dropout(x_264, 0.0, False, False)
        x_264 = None
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_267 = torch.nn.functional.dropout(x_266, 0.0, False, False)
        x_266 = None
        mul_75 = unsqueeze_151 * x_267
        unsqueeze_151 = x_267 = None
        x_268 = x_262 + mul_75
        x_262 = mul_75 = None
        unsqueeze_152 = l_self_modules_backbone_modules_network_modules_4_modules_22_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_22_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_153 = unsqueeze_152.unsqueeze(-1)
        unsqueeze_152 = None
        group_norm_78 = torch.nn.functional.group_norm(
            x_268,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_38 = torch._C._nn.avg_pool2d(
            group_norm_78, 3, 1, 1, False, False, None
        )
        sub_38 = avg_pool2d_38 - group_norm_78
        avg_pool2d_38 = group_norm_78 = None
        mul_76 = unsqueeze_153 * sub_38
        unsqueeze_153 = sub_38 = None
        x_269 = x_268 + mul_76
        x_268 = mul_76 = None
        unsqueeze_154 = l_self_modules_backbone_modules_network_modules_4_modules_22_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_22_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_155 = unsqueeze_154.unsqueeze(-1)
        unsqueeze_154 = None
        group_norm_79 = torch.nn.functional.group_norm(
            x_269,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_22_modules_norm2_parameters_bias_ = (None)
        x_270 = torch.conv2d(
            group_norm_79,
            l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_79 = l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_271 = torch._C._nn.gelu(x_270, approximate="none")
        x_270 = None
        x_272 = torch.nn.functional.dropout(x_271, 0.0, False, False)
        x_271 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_272 = l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_274 = torch.nn.functional.dropout(x_273, 0.0, False, False)
        x_273 = None
        mul_77 = unsqueeze_155 * x_274
        unsqueeze_155 = x_274 = None
        x_275 = x_269 + mul_77
        x_269 = mul_77 = None
        unsqueeze_156 = l_self_modules_backbone_modules_network_modules_4_modules_23_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_23_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_157 = unsqueeze_156.unsqueeze(-1)
        unsqueeze_156 = None
        group_norm_80 = torch.nn.functional.group_norm(
            x_275,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_39 = torch._C._nn.avg_pool2d(
            group_norm_80, 3, 1, 1, False, False, None
        )
        sub_39 = avg_pool2d_39 - group_norm_80
        avg_pool2d_39 = group_norm_80 = None
        mul_78 = unsqueeze_157 * sub_39
        unsqueeze_157 = sub_39 = None
        x_276 = x_275 + mul_78
        x_275 = mul_78 = None
        unsqueeze_158 = l_self_modules_backbone_modules_network_modules_4_modules_23_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_4_modules_23_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_159 = unsqueeze_158.unsqueeze(-1)
        unsqueeze_158 = None
        group_norm_81 = torch.nn.functional.group_norm(
            x_276,
            1,
            l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_23_modules_norm2_parameters_bias_ = (None)
        x_277 = torch.conv2d(
            group_norm_81,
            l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_81 = l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_278 = torch._C._nn.gelu(x_277, approximate="none")
        x_277 = None
        x_279 = torch.nn.functional.dropout(x_278, 0.0, False, False)
        x_278 = None
        x_280 = torch.conv2d(
            x_279,
            l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_279 = l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_4_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_281 = torch.nn.functional.dropout(x_280, 0.0, False, False)
        x_280 = None
        mul_79 = unsqueeze_159 * x_281
        unsqueeze_159 = x_281 = None
        x_282 = x_276 + mul_79
        x_276 = mul_79 = None
        x_out_2 = torch.nn.functional.group_norm(
            x_282,
            1,
            l_self_modules_backbone_modules_norm4_parameters_weight_,
            l_self_modules_backbone_modules_norm4_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_norm4_parameters_weight_ = (
            l_self_modules_backbone_modules_norm4_parameters_bias_
        ) = None
        x_283 = torch.conv2d(
            x_282,
            l_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_282 = l_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_weight_ = l_self_modules_backbone_modules_network_modules_5_modules_proj_parameters_bias_ = (None)
        unsqueeze_160 = l_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_161 = unsqueeze_160.unsqueeze(-1)
        unsqueeze_160 = None
        group_norm_83 = torch.nn.functional.group_norm(
            x_283,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_40 = torch._C._nn.avg_pool2d(
            group_norm_83, 3, 1, 1, False, False, None
        )
        sub_40 = avg_pool2d_40 - group_norm_83
        avg_pool2d_40 = group_norm_83 = None
        mul_80 = unsqueeze_161 * sub_40
        unsqueeze_161 = sub_40 = None
        x_284 = x_283 + mul_80
        x_283 = mul_80 = None
        unsqueeze_162 = l_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_0_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_163 = unsqueeze_162.unsqueeze(-1)
        unsqueeze_162 = None
        group_norm_84 = torch.nn.functional.group_norm(
            x_284,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_norm2_parameters_bias_ = (None)
        x_285 = torch.conv2d(
            group_norm_84,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_84 = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_286 = torch._C._nn.gelu(x_285, approximate="none")
        x_285 = None
        x_287 = torch.nn.functional.dropout(x_286, 0.0, False, False)
        x_286 = None
        x_288 = torch.conv2d(
            x_287,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_287 = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_289 = torch.nn.functional.dropout(x_288, 0.0, False, False)
        x_288 = None
        mul_81 = unsqueeze_163 * x_289
        unsqueeze_163 = x_289 = None
        x_290 = x_284 + mul_81
        x_284 = mul_81 = None
        unsqueeze_164 = l_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_165 = unsqueeze_164.unsqueeze(-1)
        unsqueeze_164 = None
        group_norm_85 = torch.nn.functional.group_norm(
            x_290,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_41 = torch._C._nn.avg_pool2d(
            group_norm_85, 3, 1, 1, False, False, None
        )
        sub_41 = avg_pool2d_41 - group_norm_85
        avg_pool2d_41 = group_norm_85 = None
        mul_82 = unsqueeze_165 * sub_41
        unsqueeze_165 = sub_41 = None
        x_291 = x_290 + mul_82
        x_290 = mul_82 = None
        unsqueeze_166 = l_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_1_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_167 = unsqueeze_166.unsqueeze(-1)
        unsqueeze_166 = None
        group_norm_86 = torch.nn.functional.group_norm(
            x_291,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_norm2_parameters_bias_ = (None)
        x_292 = torch.conv2d(
            group_norm_86,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_86 = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_293 = torch._C._nn.gelu(x_292, approximate="none")
        x_292 = None
        x_294 = torch.nn.functional.dropout(x_293, 0.0, False, False)
        x_293 = None
        x_295 = torch.conv2d(
            x_294,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_294 = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_296 = torch.nn.functional.dropout(x_295, 0.0, False, False)
        x_295 = None
        mul_83 = unsqueeze_167 * x_296
        unsqueeze_167 = x_296 = None
        x_297 = x_291 + mul_83
        x_291 = mul_83 = None
        unsqueeze_168 = l_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_169 = unsqueeze_168.unsqueeze(-1)
        unsqueeze_168 = None
        group_norm_87 = torch.nn.functional.group_norm(
            x_297,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_42 = torch._C._nn.avg_pool2d(
            group_norm_87, 3, 1, 1, False, False, None
        )
        sub_42 = avg_pool2d_42 - group_norm_87
        avg_pool2d_42 = group_norm_87 = None
        mul_84 = unsqueeze_169 * sub_42
        unsqueeze_169 = sub_42 = None
        x_298 = x_297 + mul_84
        x_297 = mul_84 = None
        unsqueeze_170 = l_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_2_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_171 = unsqueeze_170.unsqueeze(-1)
        unsqueeze_170 = None
        group_norm_88 = torch.nn.functional.group_norm(
            x_298,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_norm2_parameters_bias_ = (None)
        x_299 = torch.conv2d(
            group_norm_88,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_88 = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_300 = torch._C._nn.gelu(x_299, approximate="none")
        x_299 = None
        x_301 = torch.nn.functional.dropout(x_300, 0.0, False, False)
        x_300 = None
        x_302 = torch.conv2d(
            x_301,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_301 = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_303 = torch.nn.functional.dropout(x_302, 0.0, False, False)
        x_302 = None
        mul_85 = unsqueeze_171 * x_303
        unsqueeze_171 = x_303 = None
        x_304 = x_298 + mul_85
        x_298 = mul_85 = None
        unsqueeze_172 = l_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_173 = unsqueeze_172.unsqueeze(-1)
        unsqueeze_172 = None
        group_norm_89 = torch.nn.functional.group_norm(
            x_304,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_43 = torch._C._nn.avg_pool2d(
            group_norm_89, 3, 1, 1, False, False, None
        )
        sub_43 = avg_pool2d_43 - group_norm_89
        avg_pool2d_43 = group_norm_89 = None
        mul_86 = unsqueeze_173 * sub_43
        unsqueeze_173 = sub_43 = None
        x_305 = x_304 + mul_86
        x_304 = mul_86 = None
        unsqueeze_174 = l_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_3_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_175 = unsqueeze_174.unsqueeze(-1)
        unsqueeze_174 = None
        group_norm_90 = torch.nn.functional.group_norm(
            x_305,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_norm2_parameters_bias_ = (None)
        x_306 = torch.conv2d(
            group_norm_90,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_90 = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_307 = torch._C._nn.gelu(x_306, approximate="none")
        x_306 = None
        x_308 = torch.nn.functional.dropout(x_307, 0.0, False, False)
        x_307 = None
        x_309 = torch.conv2d(
            x_308,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_308 = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_310 = torch.nn.functional.dropout(x_309, 0.0, False, False)
        x_309 = None
        mul_87 = unsqueeze_175 * x_310
        unsqueeze_175 = x_310 = None
        x_311 = x_305 + mul_87
        x_305 = mul_87 = None
        unsqueeze_176 = l_self_modules_backbone_modules_network_modules_6_modules_4_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_4_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_177 = unsqueeze_176.unsqueeze(-1)
        unsqueeze_176 = None
        group_norm_91 = torch.nn.functional.group_norm(
            x_311,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_44 = torch._C._nn.avg_pool2d(
            group_norm_91, 3, 1, 1, False, False, None
        )
        sub_44 = avg_pool2d_44 - group_norm_91
        avg_pool2d_44 = group_norm_91 = None
        mul_88 = unsqueeze_177 * sub_44
        unsqueeze_177 = sub_44 = None
        x_312 = x_311 + mul_88
        x_311 = mul_88 = None
        unsqueeze_178 = l_self_modules_backbone_modules_network_modules_6_modules_4_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_4_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_179 = unsqueeze_178.unsqueeze(-1)
        unsqueeze_178 = None
        group_norm_92 = torch.nn.functional.group_norm(
            x_312,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_4_modules_norm2_parameters_bias_ = (None)
        x_313 = torch.conv2d(
            group_norm_92,
            l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_92 = l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_314 = torch._C._nn.gelu(x_313, approximate="none")
        x_313 = None
        x_315 = torch.nn.functional.dropout(x_314, 0.0, False, False)
        x_314 = None
        x_316 = torch.conv2d(
            x_315,
            l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_315 = l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_317 = torch.nn.functional.dropout(x_316, 0.0, False, False)
        x_316 = None
        mul_89 = unsqueeze_179 * x_317
        unsqueeze_179 = x_317 = None
        x_318 = x_312 + mul_89
        x_312 = mul_89 = None
        unsqueeze_180 = l_self_modules_backbone_modules_network_modules_6_modules_5_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_5_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_181 = unsqueeze_180.unsqueeze(-1)
        unsqueeze_180 = None
        group_norm_93 = torch.nn.functional.group_norm(
            x_318,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_45 = torch._C._nn.avg_pool2d(
            group_norm_93, 3, 1, 1, False, False, None
        )
        sub_45 = avg_pool2d_45 - group_norm_93
        avg_pool2d_45 = group_norm_93 = None
        mul_90 = unsqueeze_181 * sub_45
        unsqueeze_181 = sub_45 = None
        x_319 = x_318 + mul_90
        x_318 = mul_90 = None
        unsqueeze_182 = l_self_modules_backbone_modules_network_modules_6_modules_5_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_5_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_183 = unsqueeze_182.unsqueeze(-1)
        unsqueeze_182 = None
        group_norm_94 = torch.nn.functional.group_norm(
            x_319,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_5_modules_norm2_parameters_bias_ = (None)
        x_320 = torch.conv2d(
            group_norm_94,
            l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_94 = l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_321 = torch._C._nn.gelu(x_320, approximate="none")
        x_320 = None
        x_322 = torch.nn.functional.dropout(x_321, 0.0, False, False)
        x_321 = None
        x_323 = torch.conv2d(
            x_322,
            l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_322 = l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_324 = torch.nn.functional.dropout(x_323, 0.0, False, False)
        x_323 = None
        mul_91 = unsqueeze_183 * x_324
        unsqueeze_183 = x_324 = None
        x_325 = x_319 + mul_91
        x_319 = mul_91 = None
        unsqueeze_184 = l_self_modules_backbone_modules_network_modules_6_modules_6_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_6_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_185 = unsqueeze_184.unsqueeze(-1)
        unsqueeze_184 = None
        group_norm_95 = torch.nn.functional.group_norm(
            x_325,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_46 = torch._C._nn.avg_pool2d(
            group_norm_95, 3, 1, 1, False, False, None
        )
        sub_46 = avg_pool2d_46 - group_norm_95
        avg_pool2d_46 = group_norm_95 = None
        mul_92 = unsqueeze_185 * sub_46
        unsqueeze_185 = sub_46 = None
        x_326 = x_325 + mul_92
        x_325 = mul_92 = None
        unsqueeze_186 = l_self_modules_backbone_modules_network_modules_6_modules_6_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_6_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_187 = unsqueeze_186.unsqueeze(-1)
        unsqueeze_186 = None
        group_norm_96 = torch.nn.functional.group_norm(
            x_326,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_6_modules_norm2_parameters_bias_ = (None)
        x_327 = torch.conv2d(
            group_norm_96,
            l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_96 = l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_328 = torch._C._nn.gelu(x_327, approximate="none")
        x_327 = None
        x_329 = torch.nn.functional.dropout(x_328, 0.0, False, False)
        x_328 = None
        x_330 = torch.conv2d(
            x_329,
            l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_329 = l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_331 = torch.nn.functional.dropout(x_330, 0.0, False, False)
        x_330 = None
        mul_93 = unsqueeze_187 * x_331
        unsqueeze_187 = x_331 = None
        x_332 = x_326 + mul_93
        x_326 = mul_93 = None
        unsqueeze_188 = l_self_modules_backbone_modules_network_modules_6_modules_7_parameters_layer_scale_1_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_7_parameters_layer_scale_1_ = (
            None
        )
        unsqueeze_189 = unsqueeze_188.unsqueeze(-1)
        unsqueeze_188 = None
        group_norm_97 = torch.nn.functional.group_norm(
            x_332,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm1_parameters_bias_ = (None)
        avg_pool2d_47 = torch._C._nn.avg_pool2d(
            group_norm_97, 3, 1, 1, False, False, None
        )
        sub_47 = avg_pool2d_47 - group_norm_97
        avg_pool2d_47 = group_norm_97 = None
        mul_94 = unsqueeze_189 * sub_47
        unsqueeze_189 = sub_47 = None
        x_333 = x_332 + mul_94
        x_332 = mul_94 = None
        unsqueeze_190 = l_self_modules_backbone_modules_network_modules_6_modules_7_parameters_layer_scale_2_.unsqueeze(
            -1
        )
        l_self_modules_backbone_modules_network_modules_6_modules_7_parameters_layer_scale_2_ = (
            None
        )
        unsqueeze_191 = unsqueeze_190.unsqueeze(-1)
        unsqueeze_190 = None
        group_norm_98 = torch.nn.functional.group_norm(
            x_333,
            1,
            l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_7_modules_norm2_parameters_bias_ = (None)
        x_334 = torch.conv2d(
            group_norm_98,
            l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        group_norm_98 = l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_335 = torch._C._nn.gelu(x_334, approximate="none")
        x_334 = None
        x_336 = torch.nn.functional.dropout(x_335, 0.0, False, False)
        x_335 = None
        x_337 = torch.conv2d(
            x_336,
            l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_336 = l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_network_modules_6_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_338 = torch.nn.functional.dropout(x_337, 0.0, False, False)
        x_337 = None
        mul_95 = unsqueeze_191 * x_338
        unsqueeze_191 = x_338 = None
        x_339 = x_333 + mul_95
        x_333 = mul_95 = None
        x_out_3 = torch.nn.functional.group_norm(
            x_339,
            1,
            l_self_modules_backbone_modules_norm6_parameters_weight_,
            l_self_modules_backbone_modules_norm6_parameters_bias_,
            1e-05,
        )
        x_339 = (
            l_self_modules_backbone_modules_norm6_parameters_weight_
        ) = l_self_modules_backbone_modules_norm6_parameters_bias_ = None
        x_340 = torch.conv2d(
            x_out,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out = l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_0_modules_conv_parameters_bias_ = (None)
        x_341 = torch.conv2d(
            x_out_1,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_1 = l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_1_modules_conv_parameters_bias_ = (None)
        x_342 = torch.conv2d(
            x_out_2,
            l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_out_2 = l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = l_self_modules_neck_modules_lateral_convs_modules_2_modules_conv_parameters_bias_ = (None)
        x_343 = torch.conv2d(
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
            x_343, (32, 32), None, "nearest", None
        )
        add_96 = x_342 + interpolate
        x_342 = interpolate = None
        interpolate_1 = torch.nn.functional.interpolate(
            add_96, (64, 64), None, "nearest", None
        )
        add_97 = x_341 + interpolate_1
        x_341 = interpolate_1 = None
        interpolate_2 = torch.nn.functional.interpolate(
            add_97, (128, 128), None, "nearest", None
        )
        add_98 = x_340 + interpolate_2
        x_340 = interpolate_2 = None
        x_344 = torch.conv2d(
            add_98,
            l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_98 = l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_0_modules_conv_parameters_bias_ = (None)
        x_345 = torch.conv2d(
            add_97,
            l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_97 = l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_1_modules_conv_parameters_bias_ = (None)
        x_346 = torch.conv2d(
            add_96,
            l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_96 = l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_2_modules_conv_parameters_bias_ = (None)
        x_347 = torch.conv2d(
            x_343,
            l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_,
            l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_343 = l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_weight_ = l_self_modules_neck_modules_fpn_convs_modules_3_modules_conv_parameters_bias_ = (None)
        x_348 = torch.conv2d(
            x_344,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_344 = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        x_349 = torch.nn.functional.batch_norm(
            x_348,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_348 = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        x_350 = torch.nn.functional.relu(x_349, inplace=True)
        x_349 = None
        x_351 = torch.conv2d(
            x_345,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_345 = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        x_352 = torch.nn.functional.batch_norm(
            x_351,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_351 = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        x_353 = torch.nn.functional.relu(x_352, inplace=True)
        x_352 = None
        input_1 = torch.nn.functional.interpolate(
            x_353, [128, 128], None, "bilinear", False
        )
        x_353 = None
        interpolate_4 = torch.nn.functional.interpolate(
            input_1, (128, 128), None, "bilinear", False
        )
        input_1 = None
        output = x_350 + interpolate_4
        x_350 = interpolate_4 = None
        x_354 = torch.conv2d(
            x_346,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_346 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_conv_parameters_weight_ = (None)
        x_355 = torch.nn.functional.batch_norm(
            x_354,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_354 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        x_356 = torch.nn.functional.relu(x_355, inplace=True)
        x_355 = None
        input_2 = torch.nn.functional.interpolate(
            x_356, [64, 64], None, "bilinear", False
        )
        x_356 = None
        x_357 = torch.conv2d(
            input_2,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_conv_parameters_weight_ = (None)
        x_358 = torch.nn.functional.batch_norm(
            x_357,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_357 = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_2_modules_2_modules_bn_parameters_bias_ = (None)
        x_359 = torch.nn.functional.relu(x_358, inplace=True)
        x_358 = None
        input_3 = torch.nn.functional.interpolate(
            x_359, [128, 128], None, "bilinear", False
        )
        x_359 = None
        interpolate_7 = torch.nn.functional.interpolate(
            input_3, (128, 128), None, "bilinear", False
        )
        input_3 = None
        output_1 = output + interpolate_7
        output = interpolate_7 = None
        x_360 = torch.conv2d(
            x_347,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_347 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_conv_parameters_weight_ = (None)
        x_361 = torch.nn.functional.batch_norm(
            x_360,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_360 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_0_modules_bn_parameters_bias_ = (None)
        x_362 = torch.nn.functional.relu(x_361, inplace=True)
        x_361 = None
        input_4 = torch.nn.functional.interpolate(
            x_362, [32, 32], None, "bilinear", False
        )
        x_362 = None
        x_363 = torch.conv2d(
            input_4,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_conv_parameters_weight_ = (None)
        x_364 = torch.nn.functional.batch_norm(
            x_363,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_363 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_2_modules_bn_parameters_bias_ = (None)
        x_365 = torch.nn.functional.relu(x_364, inplace=True)
        x_364 = None
        input_5 = torch.nn.functional.interpolate(
            x_365, [64, 64], None, "bilinear", False
        )
        x_365 = None
        x_366 = torch.conv2d(
            input_5,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_conv_parameters_weight_ = (None)
        x_367 = torch.nn.functional.batch_norm(
            x_366,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_366 = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_scale_heads_modules_3_modules_4_modules_bn_parameters_bias_ = (None)
        x_368 = torch.nn.functional.relu(x_367, inplace=True)
        x_367 = None
        input_6 = torch.nn.functional.interpolate(
            x_368, [128, 128], None, "bilinear", False
        )
        x_368 = None
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
