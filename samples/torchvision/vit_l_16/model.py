import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_conv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_class_token_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_parameters_pos_embedding_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_heads_modules_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_heads_modules_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_conv_proj_parameters_weight_ = (
            L_self_modules_conv_proj_parameters_weight_
        )
        l_self_modules_conv_proj_parameters_bias_ = (
            L_self_modules_conv_proj_parameters_bias_
        )
        l_self_parameters_class_token_ = L_self_parameters_class_token_
        l_self_modules_encoder_parameters_pos_embedding_ = (
            L_self_modules_encoder_parameters_pos_embedding_
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_1_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_1_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_1_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_1_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_parameters_in_proj_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_parameters_in_proj_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_parameters_in_proj_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_parameters_in_proj_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_modules_out_proj_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_modules_out_proj_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_modules_out_proj_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_modules_out_proj_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_2_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_2_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_2_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_2_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_0_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_0_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_0_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_0_parameters_bias_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_3_parameters_weight_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_3_parameters_weight_
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_3_parameters_bias_ = L_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_3_parameters_bias_
        l_self_modules_encoder_modules_ln_parameters_weight_ = (
            L_self_modules_encoder_modules_ln_parameters_weight_
        )
        l_self_modules_encoder_modules_ln_parameters_bias_ = (
            L_self_modules_encoder_modules_ln_parameters_bias_
        )
        l_self_modules_heads_modules_head_parameters_weight_ = (
            L_self_modules_heads_modules_head_parameters_weight_
        )
        l_self_modules_heads_modules_head_parameters_bias_ = (
            L_self_modules_heads_modules_head_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_conv_proj_parameters_weight_,
            l_self_modules_conv_proj_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_conv_proj_parameters_weight_
        ) = l_self_modules_conv_proj_parameters_bias_ = None
        x_1 = x.reshape(1, 1024, 196)
        x = None
        x_2 = x_1.permute(0, 2, 1)
        x_1 = None
        batch_class_token = l_self_parameters_class_token_.expand(1, -1, -1)
        l_self_parameters_class_token_ = None
        x_3 = torch.cat([batch_class_token, x_2], dim=1)
        batch_class_token = x_2 = None
        input_1 = x_3 + l_self_modules_encoder_parameters_pos_embedding_
        x_3 = l_self_modules_encoder_parameters_pos_embedding_ = None
        dropout = torch.nn.functional.dropout(input_1, 0.0, False, False)
        input_1 = None
        x_4 = torch.nn.functional.layer_norm(
            dropout,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention = torch._native_multi_head_attention(
            x_4,
            x_4,
            x_4,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_4 = l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_5 = _native_multi_head_attention[0]
        _native_multi_head_attention = None
        x_6 = torch.nn.functional.dropout(x_5, 0.0, False, False)
        x_5 = None
        x_7 = x_6 + dropout
        x_6 = dropout = None
        y = torch.nn.functional.layer_norm(
            x_7,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_bias_ = (None)
        input_2 = torch._C._nn.linear(
            y,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_bias_,
        )
        y = l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_bias_ = (None)
        input_3 = torch._C._nn.gelu(input_2, approximate="none")
        input_2 = None
        input_4 = torch.nn.functional.dropout(input_3, 0.0, False, False)
        input_3 = None
        input_5 = torch._C._nn.linear(
            input_4,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_bias_,
        )
        input_4 = l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_bias_ = (None)
        input_6 = torch.nn.functional.dropout(input_5, 0.0, False, False)
        input_5 = None
        input_7 = x_7 + input_6
        x_7 = input_6 = None
        x_8 = torch.nn.functional.layer_norm(
            input_7,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_1 = torch._native_multi_head_attention(
            x_8,
            x_8,
            x_8,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_8 = l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_9 = _native_multi_head_attention_1[0]
        _native_multi_head_attention_1 = None
        x_10 = torch.nn.functional.dropout(x_9, 0.0, False, False)
        x_9 = None
        x_11 = x_10 + input_7
        x_10 = input_7 = None
        y_1 = torch.nn.functional.layer_norm(
            x_11,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_bias_ = (None)
        input_8 = torch._C._nn.linear(
            y_1,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_bias_,
        )
        y_1 = l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_bias_ = (None)
        input_9 = torch._C._nn.gelu(input_8, approximate="none")
        input_8 = None
        input_10 = torch.nn.functional.dropout(input_9, 0.0, False, False)
        input_9 = None
        input_11 = torch._C._nn.linear(
            input_10,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_bias_,
        )
        input_10 = l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_bias_ = (None)
        input_12 = torch.nn.functional.dropout(input_11, 0.0, False, False)
        input_11 = None
        input_13 = x_11 + input_12
        x_11 = input_12 = None
        x_12 = torch.nn.functional.layer_norm(
            input_13,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_2 = torch._native_multi_head_attention(
            x_12,
            x_12,
            x_12,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_12 = l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_13 = _native_multi_head_attention_2[0]
        _native_multi_head_attention_2 = None
        x_14 = torch.nn.functional.dropout(x_13, 0.0, False, False)
        x_13 = None
        x_15 = x_14 + input_13
        x_14 = input_13 = None
        y_2 = torch.nn.functional.layer_norm(
            x_15,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_bias_ = (None)
        input_14 = torch._C._nn.linear(
            y_2,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_bias_,
        )
        y_2 = l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_bias_ = (None)
        input_15 = torch._C._nn.gelu(input_14, approximate="none")
        input_14 = None
        input_16 = torch.nn.functional.dropout(input_15, 0.0, False, False)
        input_15 = None
        input_17 = torch._C._nn.linear(
            input_16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_bias_,
        )
        input_16 = l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_bias_ = (None)
        input_18 = torch.nn.functional.dropout(input_17, 0.0, False, False)
        input_17 = None
        input_19 = x_15 + input_18
        x_15 = input_18 = None
        x_16 = torch.nn.functional.layer_norm(
            input_19,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_3 = torch._native_multi_head_attention(
            x_16,
            x_16,
            x_16,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_16 = l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_17 = _native_multi_head_attention_3[0]
        _native_multi_head_attention_3 = None
        x_18 = torch.nn.functional.dropout(x_17, 0.0, False, False)
        x_17 = None
        x_19 = x_18 + input_19
        x_18 = input_19 = None
        y_3 = torch.nn.functional.layer_norm(
            x_19,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_bias_ = (None)
        input_20 = torch._C._nn.linear(
            y_3,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_bias_,
        )
        y_3 = l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_bias_ = (None)
        input_21 = torch._C._nn.gelu(input_20, approximate="none")
        input_20 = None
        input_22 = torch.nn.functional.dropout(input_21, 0.0, False, False)
        input_21 = None
        input_23 = torch._C._nn.linear(
            input_22,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_bias_,
        )
        input_22 = l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_bias_ = (None)
        input_24 = torch.nn.functional.dropout(input_23, 0.0, False, False)
        input_23 = None
        input_25 = x_19 + input_24
        x_19 = input_24 = None
        x_20 = torch.nn.functional.layer_norm(
            input_25,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_4 = torch._native_multi_head_attention(
            x_20,
            x_20,
            x_20,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_20 = l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_21 = _native_multi_head_attention_4[0]
        _native_multi_head_attention_4 = None
        x_22 = torch.nn.functional.dropout(x_21, 0.0, False, False)
        x_21 = None
        x_23 = x_22 + input_25
        x_22 = input_25 = None
        y_4 = torch.nn.functional.layer_norm(
            x_23,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_bias_ = (None)
        input_26 = torch._C._nn.linear(
            y_4,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_bias_,
        )
        y_4 = l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_bias_ = (None)
        input_27 = torch._C._nn.gelu(input_26, approximate="none")
        input_26 = None
        input_28 = torch.nn.functional.dropout(input_27, 0.0, False, False)
        input_27 = None
        input_29 = torch._C._nn.linear(
            input_28,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_bias_,
        )
        input_28 = l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_bias_ = (None)
        input_30 = torch.nn.functional.dropout(input_29, 0.0, False, False)
        input_29 = None
        input_31 = x_23 + input_30
        x_23 = input_30 = None
        x_24 = torch.nn.functional.layer_norm(
            input_31,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_5 = torch._native_multi_head_attention(
            x_24,
            x_24,
            x_24,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_24 = l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_25 = _native_multi_head_attention_5[0]
        _native_multi_head_attention_5 = None
        x_26 = torch.nn.functional.dropout(x_25, 0.0, False, False)
        x_25 = None
        x_27 = x_26 + input_31
        x_26 = input_31 = None
        y_5 = torch.nn.functional.layer_norm(
            x_27,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_bias_ = (None)
        input_32 = torch._C._nn.linear(
            y_5,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_bias_,
        )
        y_5 = l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_bias_ = (None)
        input_33 = torch._C._nn.gelu(input_32, approximate="none")
        input_32 = None
        input_34 = torch.nn.functional.dropout(input_33, 0.0, False, False)
        input_33 = None
        input_35 = torch._C._nn.linear(
            input_34,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_bias_,
        )
        input_34 = l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_bias_ = (None)
        input_36 = torch.nn.functional.dropout(input_35, 0.0, False, False)
        input_35 = None
        input_37 = x_27 + input_36
        x_27 = input_36 = None
        x_28 = torch.nn.functional.layer_norm(
            input_37,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_6 = torch._native_multi_head_attention(
            x_28,
            x_28,
            x_28,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_28 = l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_29 = _native_multi_head_attention_6[0]
        _native_multi_head_attention_6 = None
        x_30 = torch.nn.functional.dropout(x_29, 0.0, False, False)
        x_29 = None
        x_31 = x_30 + input_37
        x_30 = input_37 = None
        y_6 = torch.nn.functional.layer_norm(
            x_31,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_bias_ = (None)
        input_38 = torch._C._nn.linear(
            y_6,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_bias_,
        )
        y_6 = l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_bias_ = (None)
        input_39 = torch._C._nn.gelu(input_38, approximate="none")
        input_38 = None
        input_40 = torch.nn.functional.dropout(input_39, 0.0, False, False)
        input_39 = None
        input_41 = torch._C._nn.linear(
            input_40,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_bias_,
        )
        input_40 = l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_bias_ = (None)
        input_42 = torch.nn.functional.dropout(input_41, 0.0, False, False)
        input_41 = None
        input_43 = x_31 + input_42
        x_31 = input_42 = None
        x_32 = torch.nn.functional.layer_norm(
            input_43,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_7 = torch._native_multi_head_attention(
            x_32,
            x_32,
            x_32,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_32 = l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_33 = _native_multi_head_attention_7[0]
        _native_multi_head_attention_7 = None
        x_34 = torch.nn.functional.dropout(x_33, 0.0, False, False)
        x_33 = None
        x_35 = x_34 + input_43
        x_34 = input_43 = None
        y_7 = torch.nn.functional.layer_norm(
            x_35,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_bias_ = (None)
        input_44 = torch._C._nn.linear(
            y_7,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_bias_,
        )
        y_7 = l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_bias_ = (None)
        input_45 = torch._C._nn.gelu(input_44, approximate="none")
        input_44 = None
        input_46 = torch.nn.functional.dropout(input_45, 0.0, False, False)
        input_45 = None
        input_47 = torch._C._nn.linear(
            input_46,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_bias_,
        )
        input_46 = l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_bias_ = (None)
        input_48 = torch.nn.functional.dropout(input_47, 0.0, False, False)
        input_47 = None
        input_49 = x_35 + input_48
        x_35 = input_48 = None
        x_36 = torch.nn.functional.layer_norm(
            input_49,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_8 = torch._native_multi_head_attention(
            x_36,
            x_36,
            x_36,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_36 = l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_37 = _native_multi_head_attention_8[0]
        _native_multi_head_attention_8 = None
        x_38 = torch.nn.functional.dropout(x_37, 0.0, False, False)
        x_37 = None
        x_39 = x_38 + input_49
        x_38 = input_49 = None
        y_8 = torch.nn.functional.layer_norm(
            x_39,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_bias_ = (None)
        input_50 = torch._C._nn.linear(
            y_8,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_bias_,
        )
        y_8 = l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_bias_ = (None)
        input_51 = torch._C._nn.gelu(input_50, approximate="none")
        input_50 = None
        input_52 = torch.nn.functional.dropout(input_51, 0.0, False, False)
        input_51 = None
        input_53 = torch._C._nn.linear(
            input_52,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_bias_,
        )
        input_52 = l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_bias_ = (None)
        input_54 = torch.nn.functional.dropout(input_53, 0.0, False, False)
        input_53 = None
        input_55 = x_39 + input_54
        x_39 = input_54 = None
        x_40 = torch.nn.functional.layer_norm(
            input_55,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_9 = torch._native_multi_head_attention(
            x_40,
            x_40,
            x_40,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_40 = l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_41 = _native_multi_head_attention_9[0]
        _native_multi_head_attention_9 = None
        x_42 = torch.nn.functional.dropout(x_41, 0.0, False, False)
        x_41 = None
        x_43 = x_42 + input_55
        x_42 = input_55 = None
        y_9 = torch.nn.functional.layer_norm(
            x_43,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_bias_ = (None)
        input_56 = torch._C._nn.linear(
            y_9,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_bias_,
        )
        y_9 = l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_bias_ = (None)
        input_57 = torch._C._nn.gelu(input_56, approximate="none")
        input_56 = None
        input_58 = torch.nn.functional.dropout(input_57, 0.0, False, False)
        input_57 = None
        input_59 = torch._C._nn.linear(
            input_58,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_bias_,
        )
        input_58 = l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_bias_ = (None)
        input_60 = torch.nn.functional.dropout(input_59, 0.0, False, False)
        input_59 = None
        input_61 = x_43 + input_60
        x_43 = input_60 = None
        x_44 = torch.nn.functional.layer_norm(
            input_61,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_10 = torch._native_multi_head_attention(
            x_44,
            x_44,
            x_44,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_44 = l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_45 = _native_multi_head_attention_10[0]
        _native_multi_head_attention_10 = None
        x_46 = torch.nn.functional.dropout(x_45, 0.0, False, False)
        x_45 = None
        x_47 = x_46 + input_61
        x_46 = input_61 = None
        y_10 = torch.nn.functional.layer_norm(
            x_47,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_bias_ = (None)
        input_62 = torch._C._nn.linear(
            y_10,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_bias_,
        )
        y_10 = l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_bias_ = (None)
        input_63 = torch._C._nn.gelu(input_62, approximate="none")
        input_62 = None
        input_64 = torch.nn.functional.dropout(input_63, 0.0, False, False)
        input_63 = None
        input_65 = torch._C._nn.linear(
            input_64,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_bias_,
        )
        input_64 = l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_bias_ = (None)
        input_66 = torch.nn.functional.dropout(input_65, 0.0, False, False)
        input_65 = None
        input_67 = x_47 + input_66
        x_47 = input_66 = None
        x_48 = torch.nn.functional.layer_norm(
            input_67,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_11 = torch._native_multi_head_attention(
            x_48,
            x_48,
            x_48,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_48 = l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_49 = _native_multi_head_attention_11[0]
        _native_multi_head_attention_11 = None
        x_50 = torch.nn.functional.dropout(x_49, 0.0, False, False)
        x_49 = None
        x_51 = x_50 + input_67
        x_50 = input_67 = None
        y_11 = torch.nn.functional.layer_norm(
            x_51,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_bias_ = (None)
        input_68 = torch._C._nn.linear(
            y_11,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_bias_,
        )
        y_11 = l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_bias_ = (None)
        input_69 = torch._C._nn.gelu(input_68, approximate="none")
        input_68 = None
        input_70 = torch.nn.functional.dropout(input_69, 0.0, False, False)
        input_69 = None
        input_71 = torch._C._nn.linear(
            input_70,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_bias_,
        )
        input_70 = l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_bias_ = (None)
        input_72 = torch.nn.functional.dropout(input_71, 0.0, False, False)
        input_71 = None
        input_73 = x_51 + input_72
        x_51 = input_72 = None
        x_52 = torch.nn.functional.layer_norm(
            input_73,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_12 = torch._native_multi_head_attention(
            x_52,
            x_52,
            x_52,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_52 = l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_53 = _native_multi_head_attention_12[0]
        _native_multi_head_attention_12 = None
        x_54 = torch.nn.functional.dropout(x_53, 0.0, False, False)
        x_53 = None
        x_55 = x_54 + input_73
        x_54 = input_73 = None
        y_12 = torch.nn.functional.layer_norm(
            x_55,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_ln_2_parameters_bias_ = (None)
        input_74 = torch._C._nn.linear(
            y_12,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_0_parameters_bias_,
        )
        y_12 = l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_0_parameters_bias_ = (None)
        input_75 = torch._C._nn.gelu(input_74, approximate="none")
        input_74 = None
        input_76 = torch.nn.functional.dropout(input_75, 0.0, False, False)
        input_75 = None
        input_77 = torch._C._nn.linear(
            input_76,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_3_parameters_bias_,
        )
        input_76 = l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_12_modules_mlp_modules_3_parameters_bias_ = (None)
        input_78 = torch.nn.functional.dropout(input_77, 0.0, False, False)
        input_77 = None
        input_79 = x_55 + input_78
        x_55 = input_78 = None
        x_56 = torch.nn.functional.layer_norm(
            input_79,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_13 = torch._native_multi_head_attention(
            x_56,
            x_56,
            x_56,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_56 = l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_57 = _native_multi_head_attention_13[0]
        _native_multi_head_attention_13 = None
        x_58 = torch.nn.functional.dropout(x_57, 0.0, False, False)
        x_57 = None
        x_59 = x_58 + input_79
        x_58 = input_79 = None
        y_13 = torch.nn.functional.layer_norm(
            x_59,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_ln_2_parameters_bias_ = (None)
        input_80 = torch._C._nn.linear(
            y_13,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_0_parameters_bias_,
        )
        y_13 = l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_0_parameters_bias_ = (None)
        input_81 = torch._C._nn.gelu(input_80, approximate="none")
        input_80 = None
        input_82 = torch.nn.functional.dropout(input_81, 0.0, False, False)
        input_81 = None
        input_83 = torch._C._nn.linear(
            input_82,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_3_parameters_bias_,
        )
        input_82 = l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_13_modules_mlp_modules_3_parameters_bias_ = (None)
        input_84 = torch.nn.functional.dropout(input_83, 0.0, False, False)
        input_83 = None
        input_85 = x_59 + input_84
        x_59 = input_84 = None
        x_60 = torch.nn.functional.layer_norm(
            input_85,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_14 = torch._native_multi_head_attention(
            x_60,
            x_60,
            x_60,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_60 = l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_61 = _native_multi_head_attention_14[0]
        _native_multi_head_attention_14 = None
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        x_63 = x_62 + input_85
        x_62 = input_85 = None
        y_14 = torch.nn.functional.layer_norm(
            x_63,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_ln_2_parameters_bias_ = (None)
        input_86 = torch._C._nn.linear(
            y_14,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_0_parameters_bias_,
        )
        y_14 = l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_0_parameters_bias_ = (None)
        input_87 = torch._C._nn.gelu(input_86, approximate="none")
        input_86 = None
        input_88 = torch.nn.functional.dropout(input_87, 0.0, False, False)
        input_87 = None
        input_89 = torch._C._nn.linear(
            input_88,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_3_parameters_bias_,
        )
        input_88 = l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_14_modules_mlp_modules_3_parameters_bias_ = (None)
        input_90 = torch.nn.functional.dropout(input_89, 0.0, False, False)
        input_89 = None
        input_91 = x_63 + input_90
        x_63 = input_90 = None
        x_64 = torch.nn.functional.layer_norm(
            input_91,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_15 = torch._native_multi_head_attention(
            x_64,
            x_64,
            x_64,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_64 = l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_65 = _native_multi_head_attention_15[0]
        _native_multi_head_attention_15 = None
        x_66 = torch.nn.functional.dropout(x_65, 0.0, False, False)
        x_65 = None
        x_67 = x_66 + input_91
        x_66 = input_91 = None
        y_15 = torch.nn.functional.layer_norm(
            x_67,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_ln_2_parameters_bias_ = (None)
        input_92 = torch._C._nn.linear(
            y_15,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_0_parameters_bias_,
        )
        y_15 = l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_0_parameters_bias_ = (None)
        input_93 = torch._C._nn.gelu(input_92, approximate="none")
        input_92 = None
        input_94 = torch.nn.functional.dropout(input_93, 0.0, False, False)
        input_93 = None
        input_95 = torch._C._nn.linear(
            input_94,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_3_parameters_bias_,
        )
        input_94 = l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_15_modules_mlp_modules_3_parameters_bias_ = (None)
        input_96 = torch.nn.functional.dropout(input_95, 0.0, False, False)
        input_95 = None
        input_97 = x_67 + input_96
        x_67 = input_96 = None
        x_68 = torch.nn.functional.layer_norm(
            input_97,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_16 = torch._native_multi_head_attention(
            x_68,
            x_68,
            x_68,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_68 = l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_69 = _native_multi_head_attention_16[0]
        _native_multi_head_attention_16 = None
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        x_71 = x_70 + input_97
        x_70 = input_97 = None
        y_16 = torch.nn.functional.layer_norm(
            x_71,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_ln_2_parameters_bias_ = (None)
        input_98 = torch._C._nn.linear(
            y_16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_0_parameters_bias_,
        )
        y_16 = l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_0_parameters_bias_ = (None)
        input_99 = torch._C._nn.gelu(input_98, approximate="none")
        input_98 = None
        input_100 = torch.nn.functional.dropout(input_99, 0.0, False, False)
        input_99 = None
        input_101 = torch._C._nn.linear(
            input_100,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_3_parameters_bias_,
        )
        input_100 = l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_16_modules_mlp_modules_3_parameters_bias_ = (None)
        input_102 = torch.nn.functional.dropout(input_101, 0.0, False, False)
        input_101 = None
        input_103 = x_71 + input_102
        x_71 = input_102 = None
        x_72 = torch.nn.functional.layer_norm(
            input_103,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_17 = torch._native_multi_head_attention(
            x_72,
            x_72,
            x_72,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_72 = l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_73 = _native_multi_head_attention_17[0]
        _native_multi_head_attention_17 = None
        x_74 = torch.nn.functional.dropout(x_73, 0.0, False, False)
        x_73 = None
        x_75 = x_74 + input_103
        x_74 = input_103 = None
        y_17 = torch.nn.functional.layer_norm(
            x_75,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_ln_2_parameters_bias_ = (None)
        input_104 = torch._C._nn.linear(
            y_17,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_0_parameters_bias_,
        )
        y_17 = l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_0_parameters_bias_ = (None)
        input_105 = torch._C._nn.gelu(input_104, approximate="none")
        input_104 = None
        input_106 = torch.nn.functional.dropout(input_105, 0.0, False, False)
        input_105 = None
        input_107 = torch._C._nn.linear(
            input_106,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_3_parameters_bias_,
        )
        input_106 = l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_17_modules_mlp_modules_3_parameters_bias_ = (None)
        input_108 = torch.nn.functional.dropout(input_107, 0.0, False, False)
        input_107 = None
        input_109 = x_75 + input_108
        x_75 = input_108 = None
        x_76 = torch.nn.functional.layer_norm(
            input_109,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_18 = torch._native_multi_head_attention(
            x_76,
            x_76,
            x_76,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_76 = l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_77 = _native_multi_head_attention_18[0]
        _native_multi_head_attention_18 = None
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        x_79 = x_78 + input_109
        x_78 = input_109 = None
        y_18 = torch.nn.functional.layer_norm(
            x_79,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_ln_2_parameters_bias_ = (None)
        input_110 = torch._C._nn.linear(
            y_18,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_0_parameters_bias_,
        )
        y_18 = l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_0_parameters_bias_ = (None)
        input_111 = torch._C._nn.gelu(input_110, approximate="none")
        input_110 = None
        input_112 = torch.nn.functional.dropout(input_111, 0.0, False, False)
        input_111 = None
        input_113 = torch._C._nn.linear(
            input_112,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_3_parameters_bias_,
        )
        input_112 = l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_18_modules_mlp_modules_3_parameters_bias_ = (None)
        input_114 = torch.nn.functional.dropout(input_113, 0.0, False, False)
        input_113 = None
        input_115 = x_79 + input_114
        x_79 = input_114 = None
        x_80 = torch.nn.functional.layer_norm(
            input_115,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_19 = torch._native_multi_head_attention(
            x_80,
            x_80,
            x_80,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_80 = l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_81 = _native_multi_head_attention_19[0]
        _native_multi_head_attention_19 = None
        x_82 = torch.nn.functional.dropout(x_81, 0.0, False, False)
        x_81 = None
        x_83 = x_82 + input_115
        x_82 = input_115 = None
        y_19 = torch.nn.functional.layer_norm(
            x_83,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_ln_2_parameters_bias_ = (None)
        input_116 = torch._C._nn.linear(
            y_19,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_0_parameters_bias_,
        )
        y_19 = l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_0_parameters_bias_ = (None)
        input_117 = torch._C._nn.gelu(input_116, approximate="none")
        input_116 = None
        input_118 = torch.nn.functional.dropout(input_117, 0.0, False, False)
        input_117 = None
        input_119 = torch._C._nn.linear(
            input_118,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_3_parameters_bias_,
        )
        input_118 = l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_19_modules_mlp_modules_3_parameters_bias_ = (None)
        input_120 = torch.nn.functional.dropout(input_119, 0.0, False, False)
        input_119 = None
        input_121 = x_83 + input_120
        x_83 = input_120 = None
        x_84 = torch.nn.functional.layer_norm(
            input_121,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_20 = torch._native_multi_head_attention(
            x_84,
            x_84,
            x_84,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_84 = l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_85 = _native_multi_head_attention_20[0]
        _native_multi_head_attention_20 = None
        x_86 = torch.nn.functional.dropout(x_85, 0.0, False, False)
        x_85 = None
        x_87 = x_86 + input_121
        x_86 = input_121 = None
        y_20 = torch.nn.functional.layer_norm(
            x_87,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_ln_2_parameters_bias_ = (None)
        input_122 = torch._C._nn.linear(
            y_20,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_0_parameters_bias_,
        )
        y_20 = l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_0_parameters_bias_ = (None)
        input_123 = torch._C._nn.gelu(input_122, approximate="none")
        input_122 = None
        input_124 = torch.nn.functional.dropout(input_123, 0.0, False, False)
        input_123 = None
        input_125 = torch._C._nn.linear(
            input_124,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_3_parameters_bias_,
        )
        input_124 = l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_20_modules_mlp_modules_3_parameters_bias_ = (None)
        input_126 = torch.nn.functional.dropout(input_125, 0.0, False, False)
        input_125 = None
        input_127 = x_87 + input_126
        x_87 = input_126 = None
        x_88 = torch.nn.functional.layer_norm(
            input_127,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_21 = torch._native_multi_head_attention(
            x_88,
            x_88,
            x_88,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_88 = l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_89 = _native_multi_head_attention_21[0]
        _native_multi_head_attention_21 = None
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        x_91 = x_90 + input_127
        x_90 = input_127 = None
        y_21 = torch.nn.functional.layer_norm(
            x_91,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_ln_2_parameters_bias_ = (None)
        input_128 = torch._C._nn.linear(
            y_21,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_0_parameters_bias_,
        )
        y_21 = l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_0_parameters_bias_ = (None)
        input_129 = torch._C._nn.gelu(input_128, approximate="none")
        input_128 = None
        input_130 = torch.nn.functional.dropout(input_129, 0.0, False, False)
        input_129 = None
        input_131 = torch._C._nn.linear(
            input_130,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_3_parameters_bias_,
        )
        input_130 = l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_21_modules_mlp_modules_3_parameters_bias_ = (None)
        input_132 = torch.nn.functional.dropout(input_131, 0.0, False, False)
        input_131 = None
        input_133 = x_91 + input_132
        x_91 = input_132 = None
        x_92 = torch.nn.functional.layer_norm(
            input_133,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_22 = torch._native_multi_head_attention(
            x_92,
            x_92,
            x_92,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_92 = l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_93 = _native_multi_head_attention_22[0]
        _native_multi_head_attention_22 = None
        x_94 = torch.nn.functional.dropout(x_93, 0.0, False, False)
        x_93 = None
        x_95 = x_94 + input_133
        x_94 = input_133 = None
        y_22 = torch.nn.functional.layer_norm(
            x_95,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_ln_2_parameters_bias_ = (None)
        input_134 = torch._C._nn.linear(
            y_22,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_0_parameters_bias_,
        )
        y_22 = l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_0_parameters_bias_ = (None)
        input_135 = torch._C._nn.gelu(input_134, approximate="none")
        input_134 = None
        input_136 = torch.nn.functional.dropout(input_135, 0.0, False, False)
        input_135 = None
        input_137 = torch._C._nn.linear(
            input_136,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_3_parameters_bias_,
        )
        input_136 = l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_22_modules_mlp_modules_3_parameters_bias_ = (None)
        input_138 = torch.nn.functional.dropout(input_137, 0.0, False, False)
        input_137 = None
        input_139 = x_95 + input_138
        x_95 = input_138 = None
        x_96 = torch.nn.functional.layer_norm(
            input_139,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_1_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_1_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_1_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_1_parameters_bias_ = (None)
        _native_multi_head_attention_23 = torch._native_multi_head_attention(
            x_96,
            x_96,
            x_96,
            1024,
            16,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_parameters_in_proj_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_parameters_in_proj_bias_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_modules_out_proj_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_modules_out_proj_parameters_bias_,
            None,
            False,
            True,
            None,
        )
        x_96 = l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_parameters_in_proj_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_parameters_in_proj_bias_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_modules_out_proj_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_self_attention_modules_out_proj_parameters_bias_ = (None)
        x_97 = _native_multi_head_attention_23[0]
        _native_multi_head_attention_23 = None
        x_98 = torch.nn.functional.dropout(x_97, 0.0, False, False)
        x_97 = None
        x_99 = x_98 + input_139
        x_98 = input_139 = None
        y_23 = torch.nn.functional.layer_norm(
            x_99,
            (1024,),
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_2_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_2_parameters_bias_,
            1e-06,
        )
        l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_2_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_ln_2_parameters_bias_ = (None)
        input_140 = torch._C._nn.linear(
            y_23,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_0_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_0_parameters_bias_,
        )
        y_23 = l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_0_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_0_parameters_bias_ = (None)
        input_141 = torch._C._nn.gelu(input_140, approximate="none")
        input_140 = None
        input_142 = torch.nn.functional.dropout(input_141, 0.0, False, False)
        input_141 = None
        input_143 = torch._C._nn.linear(
            input_142,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_3_parameters_weight_,
            l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_3_parameters_bias_,
        )
        input_142 = l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_3_parameters_weight_ = l_self_modules_encoder_modules_layers_modules_encoder_layer_23_modules_mlp_modules_3_parameters_bias_ = (None)
        input_144 = torch.nn.functional.dropout(input_143, 0.0, False, False)
        input_143 = None
        input_145 = x_99 + input_144
        x_99 = input_144 = None
        x_100 = torch.nn.functional.layer_norm(
            input_145,
            (1024,),
            l_self_modules_encoder_modules_ln_parameters_weight_,
            l_self_modules_encoder_modules_ln_parameters_bias_,
            1e-06,
        )
        input_145 = (
            l_self_modules_encoder_modules_ln_parameters_weight_
        ) = l_self_modules_encoder_modules_ln_parameters_bias_ = None
        x_101 = x_100[(slice(None, None, None), 0)]
        x_100 = None
        input_146 = torch._C._nn.linear(
            x_101,
            l_self_modules_heads_modules_head_parameters_weight_,
            l_self_modules_heads_modules_head_parameters_bias_,
        )
        x_101 = (
            l_self_modules_heads_modules_head_parameters_weight_
        ) = l_self_modules_heads_modules_head_parameters_bias_ = None
        return (input_146,)
